from collections.abc import Callable

import torch
from torch import Tensor

from modules.decoding.base import BeamSearchStrategy
from modules.decoding.gumbel import GumbelTopKBeamSearch
from modules.decoding.vanilla import VanillaBeamSearch


class HybridBeamSearch(BeamSearchStrategy):
    """Hybrid beam search combining deterministic (vanilla) and stochastic (Gumbel) slots.

    Uses VanillaBeamSearch for the first k_det beam slots and GumbelTopKBeamSearch
    for the remaining k_stoch = k - k_det slots. Candidates from both are merged,
    validity-masked, deduplicated by SID tuple, and the top-k are retained.

    Args:
        k_det:  Number of deterministic (vanilla) beam slots (default: k//2).
        tau:    Gumbel temperature for the stochastic slots (default: 1.0).
    """

    def __init__(self, k_det: int | None = None, tau: float = 1.0):
        self.k_det = k_det
        self.tau = tau
        self._vanilla = VanillaBeamSearch()
        self._gumbel = GumbelTopKBeamSearch(tau=tau)

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
        if beams is None:
            # h=0: single probas [B, vocab], no split needed yet
            B = probas.size(0)
            k_det = self.k_det if self.k_det is not None else None
            # We'll generate n_cands*2 candidates total and pick best k
            # Use vanilla sampling for the first pass
            samples_v = torch.multinomial(probas, num_samples=n_cands)
            samp_log_p_v = torch.log(torch.gather(probas, 1, samples_v))

            # Gumbel-perturbed sampling for additional candidates
            from distributions.gumbel import sample_gumbel
            log_p = torch.log(probas.clamp(min=1e-10))
            perturbed = log_p / self.tau + sample_gumbel(log_p.shape, log_p.device)
            _top_vals, samples_g = perturbed.topk(n_cands, dim=-1)
            samp_log_p_g = torch.log(torch.gather(probas, 1, samples_g).clamp(min=1e-10))

            # Combine candidates
            all_samples = torch.cat([samples_v, samples_g], dim=1)    # [B, 2*n_cands]
            all_log_p = torch.cat([samp_log_p_v, samp_log_p_g], dim=1)

            is_valid = check_valid_fn(all_samples.reshape(-1, 1)).reshape(B, 2 * n_cands)
            scores, idx = all_log_p.masked_fill(~is_valid, float("-inf")).sort(
                -1, descending=True
            )

            # Infer k from log_probas if provided, otherwise use n_cands
            k = log_probas.size(1) if log_probas.dim() == 2 else n_cands

            # Deduplicate by token value
            top_tokens = torch.gather(all_samples, 1, idx)  # [B, 2*n_cands] sorted by score
            top_scores = scores  # [B, 2*n_cands]
            # Keep first occurrence of each token (highest-scoring unique tokens)
            final_tokens_list = []
            final_scores_list = []
            for b in range(B):
                seen = set()
                toks = []
                scrs = []
                for i in range(top_tokens.size(1)):
                    t = top_tokens[b, i].item()
                    if t not in seen:
                        seen.add(t)
                        toks.append(top_tokens[b, i])
                        scrs.append(top_scores[b, i])
                    if len(toks) == k:
                        break
                # Pad with -inf if fewer than k unique valid tokens found
                while len(toks) < k:
                    toks.append(torch.tensor(0, device=probas.device))
                    scrs.append(torch.tensor(float("-inf"), device=probas.device))
                final_tokens_list.append(torch.stack(toks))
                final_scores_list.append(torch.stack(scrs))

            new_beams = torch.stack(final_tokens_list, dim=0).unsqueeze(-1)  # [B, k, 1]
            new_log_probas = torch.stack(final_scores_list, dim=0)           # [B, k]
            parent_global_idx = torch.empty(0, dtype=torch.long, device=probas.device)
        else:
            # h > 0: probas is [B*k, vocab], beams is [B, k, h]
            B, k, _ = beams.shape
            k_det = self.k_det if self.k_det is not None else k // 2
            k_det = min(k_det, k)
            # --- Deterministic (vanilla) candidates ---
            samples_v = torch.multinomial(probas, num_samples=n_cands)
            samp_log_p_v = torch.log(torch.gather(probas, 1, samples_v))

            # --- Stochastic (Gumbel) candidates ---
            from distributions.gumbel import sample_gumbel
            log_p = torch.log(probas.clamp(min=1e-10))
            perturbed = log_p / self.tau + sample_gumbel(log_p.shape, log_p.device)
            _top_vals, samples_g = perturbed.topk(n_cands, dim=-1)
            samp_log_p_g = torch.log(torch.gather(probas, 1, samples_g).clamp(min=1e-10))

            # Combine all candidates: [B*k, 2*n_cands]
            all_samples = torch.cat([samples_v, samples_g], dim=1)
            all_samp_log_p = torch.cat([samp_log_p_v, samp_log_p_g], dim=1)
            total_cands = 2 * n_cands

            prev = beams.reshape(-1, h).repeat_interleave(total_cands, dim=0)
            prefix = torch.cat([prev, all_samples.reshape(-1, 1)], dim=1)
            is_valid = check_valid_fn(prefix).reshape(B, k * total_cands)

            cum_scores = (
                all_samp_log_p.reshape(B, k * total_cands)
                + log_probas.repeat_interleave(total_cands, dim=1)
            ).masked_fill(~is_valid, float("-inf"))

            # Sort by score descending for deduplication
            scores_sorted, idx_sorted = cum_scores.sort(-1, descending=True)

            # Deduplicate by SID tuple (beam_idx, token)
            all_beam_idx = idx_sorted // total_cands        # [B, k*total_cands]
            all_token_idx = torch.gather(
                all_samples.reshape(B, k * total_cands), 1, idx_sorted
            )

            # Build parent references
            final_parent_beam_list = []
            final_token_list = []
            final_score_list = []

            for b in range(B):
                seen_pairs: set[tuple[int | float, int | float]] = set()
                parents = []
                tokens = []
                scrs = []
                for i in range(all_beam_idx.size(1)):
                    pb = all_beam_idx[b, i].item()
                    tok = all_token_idx[b, i].item()
                    # SID tuple: previous beam tokens + new token
                    key = (pb, tok)
                    if key not in seen_pairs:
                        seen_pairs.add(key)
                        parents.append(all_beam_idx[b, i])
                        tokens.append(all_token_idx[b, i])
                        scrs.append(scores_sorted[b, i])
                    if len(parents) == k:
                        break
                while len(parents) < k:
                    parents.append(torch.tensor(0, device=probas.device))
                    tokens.append(torch.tensor(0, device=probas.device))
                    scrs.append(torch.tensor(float("-inf"), device=probas.device))
                final_parent_beam_list.append(torch.stack(parents))
                final_token_list.append(torch.stack(tokens))
                final_score_list.append(torch.stack(scrs))

            parent_beam_idx = torch.stack(final_parent_beam_list, dim=0)  # [B, k]
            top_tokens = torch.stack(final_token_list, dim=0)              # [B, k]
            new_log_probas = torch.stack(final_score_list, dim=0)          # [B, k]

            parent_global_idx = (
                parent_beam_idx
                + torch.arange(B, device=parent_beam_idx.device).unsqueeze(1) * k
            ).flatten()

            parent_ids = torch.gather(
                beams, 1, parent_beam_idx.unsqueeze(-1).expand(-1, -1, h)
            )
            new_ids = top_tokens.unsqueeze(-1)
            new_beams = torch.cat([parent_ids, new_ids], dim=-1)  # [B, k, h+1]

        return new_beams, new_log_probas, parent_global_idx
