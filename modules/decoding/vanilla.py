from typing import Callable, List, Optional, Tuple
import torch
from torch import Tensor
from transformers.cache_utils import DynamicCache, EncoderDecoderCache

from modules.decoding.base import BeamSearchStrategy


class VanillaBeamSearch(BeamSearchStrategy):
    """Exact replica of the original generate() sampling logic in model.py.

    At h=0: beams is None, probas is [B, vocab].
    At h>0: beams is [B, k, h], probas is [B*k, vocab].

    Returns (new_beams [B, k, h+1], new_log_probas [B, k], parent_global_idx [B*k]).
    parent_global_idx is a zero-length tensor at h=0 (no KV cache reorder needed).
    """

    def expand(
        self,
        beams: Optional[Tensor],
        log_probas: Tensor,
        probas: Tensor,
        h: int,
        n_cands: int,
        check_valid_fn: Callable,
        codebook_embs: List[Tensor],
        decoder_hidden: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        samples = torch.multinomial(probas, num_samples=n_cands)
        samp_log_p = torch.log(torch.gather(probas, 1, samples))

        if beams is None:
            # h == 0: probas is [B, vocab]
            B = probas.size(0)
            k = log_probas.size(1) if log_probas.dim() == 2 else None

            is_valid = check_valid_fn(samples.reshape(-1, 1)).reshape(B, n_cands)
            scores, idx = samp_log_p.masked_fill(~is_valid, float("-inf")).sort(
                -1, descending=True
            )
            # Infer k from scores shape if not known from log_probas
            if k is None:
                k = scores.size(1)
            top_k_idx = idx[:, :k]
            new_beams = torch.gather(samples, 1, top_k_idx).unsqueeze(-1)  # [B, k, 1]
            new_log_probas = scores[:, :k]
            # No parent reorder at h=0
            parent_global_idx = torch.empty(0, dtype=torch.long, device=probas.device)
        else:
            # h > 0: probas is [B*k, vocab], beams is [B, k, h]
            B, k, _ = beams.shape

            prev = beams.reshape(-1, h).repeat_interleave(n_cands, dim=0)
            prefix = torch.cat([prev, samples.reshape(-1, 1)], dim=1)
            is_valid = check_valid_fn(prefix).reshape(B, k * n_cands)
            scores, idx = (
                (
                    samp_log_p.reshape(B, k * n_cands)
                    + log_probas.repeat_interleave(n_cands, dim=1)
                )
                .masked_fill(~is_valid, float("-inf"))
                .sort(-1, descending=True)
            )

            top_k_idx = idx[:, :k]
            parent_beam_idx = top_k_idx // n_cands
            parent_global_idx = (
                parent_beam_idx
                + torch.arange(B, device=parent_beam_idx.device).unsqueeze(1) * k
            ).flatten()

            parent_ids = torch.gather(
                beams, 1, parent_beam_idx.unsqueeze(-1).expand(-1, -1, h)
            )
            new_ids = torch.gather(
                samples.reshape(B, k * n_cands), 1, top_k_idx
            ).unsqueeze(-1)
            new_beams = torch.cat([parent_ids, new_ids], dim=-1)  # [B, k, h+1]
            new_log_probas = scores[:, :k]

        return new_beams, new_log_probas, parent_global_idx
