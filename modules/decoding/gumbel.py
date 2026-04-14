from collections.abc import Callable

import torch
from torch import Tensor

from distributions.gumbel import sample_gumbel
from modules.decoding.base import BeamSearchStrategy


class GumbelTopKBeamSearch(BeamSearchStrategy):
    """Beam search using Gumbel top-k perturbation instead of multinomial sampling.

    Replaces torch.multinomial with deterministic top-k over Gumbel-perturbed log-probs.
    True log-probs are still used for cumulative scoring.

    Args:
        tau: Temperature for Gumbel perturbation. Higher tau => more diversity.
    """

    def __init__(self, tau: float = 1.0):
        self.tau = tau

    def expand(
        self,
        beams: Tensor | None,
        log_probas: Tensor,
        probas: Tensor,
        h: int,
        n_cands: int,
        check_valid_fn: Callable,
        codebook_embs: list[Tensor],
        decoder_hidden: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        log_p = torch.log(probas.clamp(min=1e-10))
        perturbed = log_p / self.tau + sample_gumbel(log_p.shape, log_p.device)
        # Take top n_cands from perturbed scores (deterministic given noise)
        _top_vals, samples = perturbed.topk(n_cands, dim=-1)
        # Use true log-probs for cumulative scoring
        samp_log_p = torch.log(torch.gather(probas, 1, samples).clamp(min=1e-10))

        if beams is None:
            B = probas.size(0)
            k = log_probas.size(1) if log_probas.dim() == 2 else None

            is_valid = check_valid_fn(samples.reshape(-1, 1)).reshape(B, n_cands)
            scores, idx = samp_log_p.masked_fill(~is_valid, float("-inf")).sort(
                -1, descending=True
            )
            if k is None:
                k = scores.size(1)
            top_k_idx = idx[:, :k]
            new_beams = torch.gather(samples, 1, top_k_idx).unsqueeze(-1)  # [B, k, 1]
            new_log_probas = scores[:, :k]
            parent_global_idx = torch.empty(0, dtype=torch.long, device=probas.device)
        else:
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
