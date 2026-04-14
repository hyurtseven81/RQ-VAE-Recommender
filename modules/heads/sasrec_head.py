
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SASRecAuxHead(nn.Module):
    """SASRec-style auxiliary head for hybrid decoding.

    Projects decoder hidden state (384-d) to item query vector (32-d),
    scored via dot product against a learned item embedding table initialized
    from frozen RQ-VAE encoder outputs.

    Architectural distinction from LIGER: LIGER projects to frozen Sentence-T5
    TEXT embeddings (content signal, 768-d). We project to a LEARNED item
    embedding table initialized from RQ-VAE encoder outputs (collaborative
    signal, 32-d). Both the signal source and dimensionality differ.
    """

    def __init__(
        self,
        d_model: int,           # decoder hidden dim (384 for Amazon configs)
        d_item: int,            # item embedding dim = RQ-VAE embed_dim (32)
        num_items: int,         # corpus size
        dropout: float = 0.1,
        item_embeddings_init: Tensor | None = None,  # [num_items, d_item] from RQ-VAE encoder
    ):
        super().__init__()
        self.d_item = d_item

        # 2-layer MLP: d_model -> d_model -> d_item
        self.norm = nn.LayerNorm(d_model)
        self.proj1 = nn.Linear(d_model, d_model, bias=True)
        self.proj2 = nn.Linear(d_model, d_item, bias=True)
        self.dropout = nn.Dropout(dropout)

        # Item embedding table [num_items, d_item]
        # Initialized from RQ-VAE encoder outputs, then learned jointly
        self.item_embeddings = nn.Embedding(num_items, d_item)
        if item_embeddings_init is not None:
            with torch.no_grad():
                self.item_embeddings.weight.copy_(item_embeddings_init)

    def forward(self, decoder_hidden: Tensor) -> Tensor:
        """
        Args:
            decoder_hidden: [B, d_model] -- final decoder hidden state
        Returns:
            query: [B, d_item] -- query vector for dot-product scoring
        """
        x = self.norm(decoder_hidden)
        x = F.gelu(self.proj1(x))
        x = self.dropout(x)
        query = self.proj2(x)
        return query

    def score_all_items(self, query: Tensor) -> Tensor:
        """
        Args:
            query: [B, d_item]
        Returns:
            scores: [B, num_items]
        """
        return query @ self.item_embeddings.weight.T

    def score_items(self, query: Tensor, item_ids: Tensor) -> Tensor:
        """
        Args:
            query: [B, d_item]
            item_ids: [B, N] item indices
        Returns:
            scores: [B, N]
        """
        item_embs = self.item_embeddings(item_ids)  # [B, N, d_item]
        return (query.unsqueeze(1) * item_embs).sum(-1)


def build_item_embeddings_from_rqvae(
    rqvae: nn.Module,
    item_features: Tensor,  # [num_items, feature_dim]
    batch_size: int = 256,
    device: str = "cpu",
) -> Tensor:
    """Compute RQ-VAE encoder outputs for all items.

    Returns [num_items, d_item] tensor to initialize SASRecAuxHead.item_embeddings.
    """
    rqvae.eval()
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(item_features), batch_size):
            batch = item_features[i:i + batch_size].to(device)
            emb = rqvae.encode(batch)  # type: ignore[operator]  # [B, d_item]
            embeddings.append(emb.cpu())
    return torch.cat(embeddings, dim=0)
