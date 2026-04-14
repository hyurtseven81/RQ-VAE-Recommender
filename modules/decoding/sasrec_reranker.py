"""Post-hoc SASRec reranker for completed beam search outputs.

This is the H2a baseline: conceptually similar to LIGER's reranking
but using a collaborative-signal head instead of frozen text embeddings.
Unlike LevelAwareHybridDecoding (in-loop, per-level mixing), this reranker
operates AFTER full generation completes.
"""

import torch
import torch.nn as nn
from torch import Tensor


class SASRecReranker:
    """Post-hoc reranker: scores completed beams with SASRec dense signal.

    This is the H2a baseline: conceptually similar to LIGER's reranking
    but using a collaborative-signal head instead of frozen text embeddings.
    """

    def __init__(self, aux_head: nn.Module, alpha: float = 0.5):
        self.aux_head = aux_head
        self.alpha = alpha

    def rerank(
        self,
        beams: Tensor,            # [B, k, L] completed SID tuples
        log_probas: Tensor,       # [B, k] from beam search
        codebook_embs: list[Tensor],  # List[L] of [vocab, 32]
        decoder_hidden: Tensor,   # [B, d_model] final decoder hidden state
    ) -> tuple[Tensor, Tensor]:   # reranked (beams, scores)
        """
        1. Reconstruct full item embedding for each beam: r_full = sum_l e_{c_l}
        2. Compute SASRec query: q = aux_head(decoder_hidden)  [B, 32]
        3. Compute dense scores: s = (q * r_full).sum(-1) per beam
        4. Z-score normalize log_probas and dense scores across k beams
        5. Mix: score = (1-alpha)*norm_log_p + alpha*norm_s_dense
        6. Re-sort beams by mixed score
        """
        B, k, L = beams.shape

        # Reconstruct full item embedding for each beam: sum over all levels
        # beams: [B, k, L]
        # r_full: [B, k, d_item]
        r_full = sum(
            codebook_embs[lvl][beams[:, :, lvl]]  # [B, k, d_item]
            for lvl in range(L)
        )  # [B, k, d_item]

        # Compute SASRec query
        with torch.no_grad():
            q = self.aux_head(decoder_hidden)  # [B, d_item]

        # Dense scores: dot product of query with each beam's reconstruction
        # q: [B, d_item] -> [B, 1, d_item]
        # r_full: [B, k, d_item]
        s_dense = (q.unsqueeze(1) * r_full).sum(-1)  # [B, k]

        # Z-score normalize both signals across the k beams dimension
        norm_log_p = self._z_score(log_probas, dim=-1)    # [B, k]
        norm_s_dense = self._z_score(s_dense, dim=-1)     # [B, k]

        # Mix
        mixed = (1.0 - self.alpha) * norm_log_p + self.alpha * norm_s_dense  # [B, k]

        # Re-sort beams by mixed score (descending)
        _, sort_idx = mixed.sort(dim=-1, descending=True)  # [B, k]

        reranked_beams = torch.gather(
            beams, 1, sort_idx.unsqueeze(-1).expand(-1, -1, L)
        )  # [B, k, L]
        reranked_scores = torch.gather(mixed, 1, sort_idx)  # [B, k]

        return reranked_beams, reranked_scores

    def _z_score(self, x: Tensor, dim: int = -1) -> Tensor:
        """Z-score normalize along dim. Returns zeros if std is near zero."""
        eps = 1e-8
        mean = x.mean(dim=dim, keepdim=True)
        std = x.std(dim=dim, keepdim=True)
        return (x - mean) / (std + eps)
