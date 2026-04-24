from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import T5EncoderModel
from transformers.cache_utils import DynamicCache, EncoderDecoderCache
from transformers.models.t5.modeling_t5 import T5Config, T5Stack

from data.schemas import TokenizedSeqBatch
from modules.decoding.base import BeamSearchStrategy

torch.set_float32_matmul_precision("high")


class ModelOutput(NamedTuple):
    loss: Tensor
    logits: Tensor
    loss_d: Tensor


class GenerationOutput(NamedTuple):
    sem_ids: Tensor
    log_probas: Tensor
    # Optional: encoder-side query hidden state [B, d_model] produced at level 0,
    # before any beam expansion. Consumed by post-hoc reranking strategies
    # (e.g. SASRecReranker) to generate a single-query-per-user dense score.
    # None unless generate(..., return_query_hidden=True) was requested.
    query_hidden: Tensor | None = None


def _strip_dedup_col(
    tensor: torch.Tensor, sem_ids_dim: int, n_layers: int
) -> torch.Tensor:
    """Strip the deduplication column appended by SemanticIdTokenizer.

    Args:
        tensor:      [B, N * sem_ids_dim]  where sem_ids_dim = n_layers + 1
        sem_ids_dim: tokens per item including the dedup column
        n_layers:    number of RQ-VAE codebook levels

    Returns:
        [B, N * n_layers]
    """
    B, total = tensor.shape
    N = total // sem_ids_dim
    return (
        tensor.view(B, N, sem_ids_dim)[:, :, :n_layers]
        .contiguous()
        .view(B, N * n_layers)
    )


class EncoderDecoderRetrievalModel(nn.Module):
    """HuggingFace T5 encoder-decoder for sequential recommendation.

    Uses T5EncoderModel for encoding and T5Stack for decoding. Per-hierarchy
    linear output heads project decoder hidden states to codebook logits.
    Beam search uses multinomial sampling with log-probability accumulation
    and a float("-inf") mask for invalid SID prefixes.
    """

    def __init__(
        self,
        codebooks: torch.Tensor,
        num_hierarchies: int,
        num_embeddings_per_hierarchy: int,
        t5_d_model: int = 128,
        t5_num_heads: int = 6,
        t5_d_ff: int = 1024,
        t5_num_layers: int = 4,
        top_k_for_generation: int = 10,
        should_add_sep_token: bool = True,
        num_user_bins: int | None = None,
    ):
        super().__init__()

        self.num_hierarchies = num_hierarchies
        self.num_embeddings_per_hierarchy = num_embeddings_per_hierarchy
        self.top_k_for_generation = top_k_for_generation
        self.register_buffer("codebooks", codebooks)

        encoder_config = T5Config(
            vocab_size=num_embeddings_per_hierarchy * num_hierarchies,
            d_model=t5_d_model,
            num_heads=t5_num_heads,
            d_ff=t5_d_ff,
            num_layers=t5_num_layers,
            is_decoder=False,
        )
        self.encoder = T5EncoderModel(encoder_config)

        decoder_config = T5Config(
            vocab_size=num_embeddings_per_hierarchy * num_hierarchies,
            d_model=t5_d_model,
            num_heads=t5_num_heads,
            d_ff=t5_d_ff,
            num_layers=t5_num_layers,
            is_decoder=True,
            is_encoder_decoder=False,
        )
        self.t5_decoder = T5Stack(decoder_config)
        self.bos_token = nn.Parameter(torch.randn(1, t5_d_model), requires_grad=True)
        self.decoder_mlp = nn.ModuleList(
            [
                nn.Linear(t5_d_model, num_embeddings_per_hierarchy, bias=False)
                for _ in range(num_hierarchies)
            ]
        )

        # Shared embedding table; hierarchy h, token t maps to index h * codebook_size + t.
        self.item_sid_embedding_table = nn.Embedding(
            num_embeddings=num_embeddings_per_hierarchy * num_hierarchies,
            embedding_dim=t5_d_model,
        )

        self.user_embedding = (
            nn.Embedding(num_user_bins, t5_d_model) if num_user_bins else None
        )
        self.sep_token = (
            nn.Parameter(torch.randn(1, t5_d_model), requires_grad=True)
            if should_add_sep_token
            else None
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _is_cache_valid(self, kv) -> bool:
        if isinstance(kv, (EncoderDecoderCache, DynamicCache)):
            return len(kv) > 0
        return isinstance(kv, tuple)

    def _add_repeating_offset_to_rows(
        self,
        input_sids: torch.Tensor,
        codebook_size: int,
        num_hierarchies: int,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Add per-hierarchy offsets so a single embedding table covers all hierarchies."""
        if input_sids.ndim != 2:
            raise ValueError("Input tensor must be 2-dimensional.")
        _, num_cols = input_sids.shape
        offsets = (
            torch.arange(num_hierarchies, device=input_sids.device) * codebook_size
        )
        num_repeats = (num_cols + num_hierarchies - 1) // num_hierarchies
        repeated_offsets = offsets.repeat(num_repeats)[:num_cols]
        result = input_sids + repeated_offsets
        if attention_mask is not None:
            result = result * attention_mask
        return result

    def _inject_sep_token_between_sids(
        self,
        id_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        sep_token: torch.Tensor,
        num_hierarchies: int,
    ):
        """Inject a separator embedding after each item's token group."""
        batch_size, seq_len, emb_dim = id_embeddings.size()
        item_count = seq_len // num_hierarchies
        reshaped_emb = id_embeddings.view(batch_size, item_count, num_hierarchies, -1)
        reshaped_mask = attention_mask.view(batch_size, item_count, num_hierarchies)
        sep = sep_token.unsqueeze(0).expand(batch_size, item_count, -1).unsqueeze(-2)
        id_embeddings = torch.cat([reshaped_emb, sep], dim=-2)
        attention_mask = torch.cat([reshaped_mask, reshaped_mask[:, :, [-1]]], dim=-1)
        return id_embeddings.reshape(batch_size, -1, emb_dim), attention_mask.reshape(
            batch_size, -1
        )

    def _check_valid_prefix(
        self, prefix: torch.Tensor, batch_size: int = 100000
    ) -> torch.Tensor:
        """Return a boolean mask indicating which prefixes exist in the corpus codebook."""
        if prefix.device != self.codebooks.device:
            self.codebooks = self.codebooks.to(prefix.device)
        trimmed = self.codebooks[:, : prefix.shape[1]]
        results = []
        for i in range(0, prefix.shape[0], batch_size):
            batch = prefix[i : i + batch_size]
            results.append(
                (trimmed.unsqueeze(1) == batch.unsqueeze(0)).all(dim=2).any(dim=0)
            )
        return torch.cat(results)

    def encoder_forward_pass(self, attention_mask, input_ids, user_id=None):
        shifted = self._add_repeating_offset_to_rows(
            input_sids=input_ids,
            codebook_size=self.num_embeddings_per_hierarchy,
            num_hierarchies=self.num_hierarchies,
            attention_mask=attention_mask,
        )
        inputs_embeds = self.item_sid_embedding_table(shifted)

        if self.sep_token is not None:
            inputs_embeds, attention_mask = self._inject_sep_token_between_sids(
                id_embeddings=inputs_embeds,
                attention_mask=attention_mask,
                sep_token=self.sep_token,
                num_hierarchies=self.num_hierarchies,
            )

        if user_id is not None and self.user_embedding is not None:
            user_embeds = self.user_embedding(
                torch.remainder(user_id[:, 0], self.user_embedding.num_embeddings)
            )
            inputs_embeds = torch.cat([user_embeds.unsqueeze(1), inputs_embeds], dim=1)
            attention_mask = torch.cat(
                [
                    torch.ones(attention_mask.size(0), 1, device=attention_mask.device),
                    attention_mask,
                ],
                dim=1,
            )

        encoder_output = self.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        ).last_hidden_state
        return encoder_output, attention_mask

    def decoder_forward_pass(
        self,
        attention_mask=None,
        future_ids=None,
        encoder_output=None,
        attention_mask_for_encoder=None,
        use_cache=False,
        past_key_values=None,
    ):
        if future_ids is not None:
            shifted = self._add_repeating_offset_to_rows(
                input_sids=future_ids,
                codebook_size=self.num_embeddings_per_hierarchy,
                num_hierarchies=self.num_hierarchies,
                attention_mask=torch.ones_like(future_ids)
                if attention_mask is None
                else attention_mask,
            )
            inputs_embeds = self.item_sid_embedding_table(shifted)

            if not self._is_cache_valid(past_key_values):
                bos = self.bos_token.unsqueeze(0).expand(future_ids.size(0), 1, -1)
                inputs_embeds = torch.cat([bos, inputs_embeds], dim=1)
                if attention_mask is not None:
                    attention_mask = torch.cat(
                        [
                            torch.ones(future_ids.size(0), 1, device=future_ids.device),
                            attention_mask,
                        ],
                        dim=1,
                    )
            else:
                inputs_embeds = inputs_embeds[:, -1:, :]
        else:
            inputs_embeds = self.bos_token.unsqueeze(0).expand(
                encoder_output.size(0), 1, -1
            )

        out = self.t5_decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_output,
            encoder_attention_mask=attention_mask_for_encoder,
            use_cache=use_cache,
            past_key_values=past_key_values,
        )
        if use_cache:
            return out.last_hidden_state, out.past_key_values
        return out.last_hidden_state

    def forward(self, batch: TokenizedSeqBatch) -> ModelOutput:
        sem_ids_dim = self.num_hierarchies + 1
        input_ids = _strip_dedup_col(batch.sem_ids, sem_ids_dim, self.num_hierarchies)
        attention_mask = _strip_dedup_col(
            batch.seq_mask.long(), sem_ids_dim, self.num_hierarchies
        )
        fut_ids = batch.sem_ids_fut[:, : self.num_hierarchies]

        encoder_output, attention_mask_for_encoder = self.encoder_forward_pass(
            attention_mask=attention_mask,
            input_ids=input_ids,
            user_id=batch.user_ids,
        )
        decoder_output = self.decoder_forward_pass(
            future_ids=fut_ids,
            encoder_output=encoder_output,
            attention_mask_for_encoder=attention_mask_for_encoder,
            use_cache=False,
        )[:, :-1]  # [B, num_hierarchies, d_model]

        total_loss = torch.tensor(0.0, device=decoder_output.device)
        loss_d = []
        for h in range(self.num_hierarchies):
            logits = self.decoder_mlp[h](decoder_output[:, h])
            h_loss = F.cross_entropy(logits, fut_ids[:, h].long())
            total_loss = total_loss + h_loss
            loss_d.append(h_loss.detach())

        return ModelOutput(loss=total_loss, logits=None, loss_d=torch.stack(loss_d))

    @torch.no_grad()
    def generate(
        self,
        attention_mask,
        input_ids,
        user_id=None,
        strategy: BeamSearchStrategy | None = None,
        codebook_embs: list[Tensor] | None = None,
        return_query_hidden: bool = False,
    ):
        """Generate top-k semantic IDs using sampling-based beam search.

        For each hierarchy level, delegates candidate expansion to ``strategy``
        (default: :class:`VanillaBeamSearch`, which is byte-for-byte equivalent to
        the original inline logic).

        Args:
            attention_mask:  [B, seq_len]
            input_ids:       [B, seq_len]
            user_id:         Optional user ids for personalized encoding.
            strategy:        Decoding strategy instance. Defaults to VanillaBeamSearch.
            codebook_embs:   Optional list of L tensors [vocab, embed_dim] with the
                             per-level codebook embedding tables. Required by strategies
                             that use embedding-space diversity (e.g. DiverseBeamSearch
                             with use_embedding_distance=True). When None, strategies
                             that need it will raise NotImplementedError.
            return_query_hidden: If True, also returns the decoder hidden state from
                             level 0 (before any beam expansion) as a [B, d_model]
                             tensor. Consumed by post-hoc reranking strategies like
                             :class:`SASRecReranker` to produce one query per user.

        Returns:
            (generated_ids, log_probas) or, when return_query_hidden=True,
            (generated_ids, log_probas, query_hidden) where
            query_hidden is [B, d_model] captured at level 0.
        """
        if strategy is None:
            from modules.decoding.vanilla import VanillaBeamSearch
            strategy = VanillaBeamSearch()

        B = input_ids.size(0)
        k = self.top_k_for_generation
        n_cands = min(200, self.num_embeddings_per_hierarchy)

        enc_out, enc_mask = self.encoder_forward_pass(
            attention_mask=attention_mask,
            input_ids=input_ids,
            user_id=user_id,
        )
        rep_enc = enc_out.repeat_interleave(k, dim=0)
        rep_mask = enc_mask.repeat_interleave(k, dim=0)

        generated = None  # [B, k, h] grows with each hierarchy step
        log_probas = torch.zeros(B, k, device=input_ids.device)
        past_kv = EncoderDecoderCache(DynamicCache(), DynamicCache())
        query_hidden: Tensor | None = None

        for h in range(self.num_hierarchies):
            if generated is not None:
                cur_enc, cur_mask = rep_enc, rep_mask
                squeezed = generated.reshape(-1, h)
            else:
                cur_enc, cur_mask = enc_out, enc_mask
                squeezed = None

            dec_out, past_kv = self.decoder_forward_pass(
                future_ids=squeezed,
                encoder_output=cur_enc,
                attention_mask_for_encoder=cur_mask,
                use_cache=True,
                past_key_values=past_kv,
            )

            probas = F.softmax(self.decoder_mlp[h](dec_out[:, -1, :]), dim=-1)

            # Capture the level-0 decoder hidden as the "user query" for
            # post-hoc reranking. At level 0 the decoder has no beam-expanded
            # input yet, so dec_out has shape [B, seq, d_model] and the last
            # token's hidden state is a natural per-user query.
            if h == 0 and return_query_hidden:
                query_hidden = dec_out[:, -1, :].detach().clone()

            generated, log_probas, parent_global_idx = strategy.expand(
                beams=generated,
                log_probas=log_probas,
                probas=probas,
                h=h,
                n_cands=n_cands,
                check_valid_fn=self._check_valid_prefix,
                codebook_embs=codebook_embs,
                decoder_hidden=dec_out[:, -1, :],
            )

            if h == 0:
                # Reset KV cache after first level (matches original behaviour)
                past_kv = EncoderDecoderCache(DynamicCache(), DynamicCache())
            else:
                past_kv.reorder_cache(parent_global_idx)

        if return_query_hidden:
            return generated, log_probas, query_hidden
        return generated, log_probas

    @torch.no_grad()
    def generate_next_sem_id(
        self,
        batch: TokenizedSeqBatch,
        top_k: bool = True,
        temperature: int = 1,
        strategy: BeamSearchStrategy | None = None,
        codebook_embs: list[Tensor] | None = None,
        return_query_hidden: bool = False,
    ) -> GenerationOutput:
        sem_ids_dim = self.num_hierarchies + 1
        input_ids = _strip_dedup_col(batch.sem_ids, sem_ids_dim, self.num_hierarchies)
        attention_mask = _strip_dedup_col(
            batch.seq_mask.long(), sem_ids_dim, self.num_hierarchies
        )
        if return_query_hidden:
            generated_ids, log_probas, query_hidden = self.generate(
                attention_mask=attention_mask,
                input_ids=input_ids,
                user_id=batch.user_ids,
                strategy=strategy,
                codebook_embs=codebook_embs,
                return_query_hidden=True,
            )
            return GenerationOutput(
                sem_ids=generated_ids,
                log_probas=log_probas,
                query_hidden=query_hidden,
            )
        generated_ids, log_probas = self.generate(
            attention_mask=attention_mask,
            input_ids=input_ids,
            user_id=batch.user_ids,
            strategy=strategy,
            codebook_embs=codebook_embs,
        )
        return GenerationOutput(sem_ids=generated_ids, log_probas=log_probas)
