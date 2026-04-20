"""Level-aware hybrid decoding: mix dense and autoregressive scores at each codebook level.

This is the primary contribution of the CIKM 2026 paper. At each codebook level l,
we mix the autoregressive log-probability with a SASRec-style dense score computed
against the partial RQ reconstruction sum_{i<=l} e_{c_i}, using a level-dependent
weight alpha_l.

Key distinction from prior work:
- LIGER: reranks completed items AFTER full generation (post-hoc)
- COBRA: BeamFusion AFTER all codebook tokens decoded (post-hoc)
- Ours: mixes PER-LEVEL DURING beam expansion (in-loop)
"""
from collections.abc import Callable

import torch
import torch.nn as nn
from torch import Tensor

from modules.decoding.base import BeamSearchStrategy


class LevelAwareHybridDecoding(BeamSearchStrategy):
    """Wraps any base BeamSearchStrategy and applies per-level alpha mixing.

    At each codebook level h:
    1. Compute base strategy's log-probs (from probas arg)
    2. Compute partial RQ reconstruction r_h^cand = sum_{i<h} e_{c_i} + e_{c_h}^cand
    3. Compute dense score: s = <q, r_h^cand> where q = aux_head(decoder_hidden)
    4. Z-score both log_p and s to unit scale (unit-free alpha interpretation)
    5. Mix in log-space: mixed = (1 - alpha[h]) * norm_log_p + alpha[h] * norm_s
    6. Restore log-prob scale (multiply by log_p's std) so softmax preserves
       the original sharpness. Without this rescale, softmax of unit-variance
       scores collapses toward uniform.
    7. Convert to probas via softmax and delegate to base strategy for
       validity masking and top-k selection.

    Composability: wraps any base strategy (Vanilla, DBS, Gumbel, Hybrid).
    """

    def __init__(
        self,
        base_strategy: BeamSearchStrategy,
        alpha: list[float],          # [alpha_0, alpha_1, ..., alpha_{L-1}]
        aux_head: nn.Module,         # SASRecAuxHead (frozen at inference)
        eps: float = 1e-8,           # for z-score stability
    ):
        self.base_strategy = base_strategy
        self.alpha = alpha
        self.aux_head = aux_head
        self.eps = eps

    def expand(
        self,
        beams: Tensor | None,         # [B, k, h] or None at h=0
        log_probas: Tensor,              # [B, k]
        probas: Tensor,                  # [B*k, vocab] or [B, vocab] at h=0
        h: int,
        n_cands: int,
        check_valid_fn: Callable,
        codebook_embs: list[Tensor],     # List[L] of [vocab, d_item=32]
        decoder_hidden: Tensor | None = None,  # [B*k, d_model] or [B, d_model] at h=0
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Algorithm:
        1. Get all candidate token IDs from the full vocab
        2. Compute partial RQ reconstruction for each (beam, candidate) pair
        3. Compute SASRec dense score via dot product with query from aux_head
        4. Z-score normalize log_p and dense scores across candidates dimension
        5. Mix with alpha[h]
        6. Convert mixed scores to probas via softmax
        7. Call base_strategy.expand() with modified probas
        """
        a = self.alpha[h] if h < len(self.alpha) else 0.0

        # If alpha=0 at this level, skip mixing entirely (exact base strategy behavior)
        if a == 0.0 or decoder_hidden is None:
            return self.base_strategy.expand(
                beams=beams,
                log_probas=log_probas,
                probas=probas,
                h=h,
                n_cands=n_cands,
                check_valid_fn=check_valid_fn,
                codebook_embs=codebook_embs,
                decoder_hidden=decoder_hidden,
            )

        vocab = probas.size(-1)

        # Compute log-probs from probas
        log_p = torch.log(probas.clamp(min=1e-10))  # [B*k, vocab] or [B, vocab]

        # Compute partial reconstruction for all candidate vocab tokens
        cand_ids = torch.arange(vocab, device=probas.device)  # [vocab]
        r = self._compute_partial_reconstruction(beams, cand_ids, codebook_embs, h)
        # r: [B, k, vocab, d] or [1, vocab, d] at h=0

        # Compute SASRec query
        with torch.no_grad():
            q = self.aux_head(decoder_hidden)  # [B*k, d_item] or [B, d_item]

        if beams is None:
            # h=0: probas is [B, vocab], q is [B, d_item]
            # r is [1, vocab, d_item] — broadcast over B
            # s_dense: [B, vocab]
            s_dense = (q.unsqueeze(1) * r).sum(-1)  # [B, vocab]

            mixed = self._mix_logspace(log_p, s_dense, a)  # [B, vocab]
        else:
            # h>0: probas is [B*k, vocab], q is [B*k, d_item]
            # r is [B, k, vocab, d_item]
            B, k, _ = beams.shape

            # Reshape q to [B, k, d_item] for broadcasting
            q_bk = q.reshape(B, k, -1)  # [B, k, d_item]

            # s_dense: [B, k, vocab]
            s_dense = (q_bk.unsqueeze(2) * r).sum(-1)  # [B, k, vocab]

            # log_p: [B*k, vocab] -> [B, k, vocab]
            log_p_bk = log_p.reshape(B, k, vocab)

            mixed = self._mix_logspace(log_p_bk, s_dense, a)  # [B, k, vocab]

            # Flatten back to [B*k, vocab] for base strategy
            mixed = mixed.reshape(B * k, vocab)

        # Convert mixed logits to probas. Because _mix_logspace restores
        # log_p's std, softmax here preserves the original distribution's
        # sharpness (fixes audit bug B2: softmax-over-unit-variance).
        mixed_probas = torch.softmax(mixed, dim=-1)

        return self.base_strategy.expand(
            beams=beams,
            log_probas=log_probas,
            probas=mixed_probas,
            h=h,
            n_cands=n_cands,
            check_valid_fn=check_valid_fn,
            codebook_embs=codebook_embs,
            decoder_hidden=decoder_hidden,
        )

    def _compute_partial_reconstruction(
        self,
        beams: Tensor | None,   # [B, k, h] — previously chosen tokens
        cand_ids: Tensor,           # [vocab] — all candidate token IDs at level h
        codebook_embs: list[Tensor],  # List[L] of [vocab, 32]
        h: int,
    ) -> Tensor:
        """Compute r_h^cand = sum_{i<h} e_{c_i} + e_{cand} for all beams and candidates.

        Returns: [B, k, vocab, 32] (or [1, vocab, 32] at h=0)
        """
        if h == 0:
            # r shape: [1, vocab, 32] (broadcast over B)
            r = codebook_embs[0]        # [vocab, 32]
            r = r.unsqueeze(0)          # [1, vocab, 32]
            return r
        else:
            assert beams is not None  # h > 0 guarantees beams exist
            # Sum previous level embeddings for each beam
            # beams: [B, k, h], codebook_embs[i]: [vocab, 32]
            past_embs: Tensor = torch.stack([
                codebook_embs[i][beams[:, :, i]]  # [B, k, 32]
                for i in range(h)
            ]).sum(0)  # [B, k, 32]
            # Add current level candidate embedding
            cand_embs = codebook_embs[h]  # [vocab, 32]
            # r: [B, k, vocab, 32]
            r = past_embs.unsqueeze(2) + cand_embs.unsqueeze(0).unsqueeze(0)
            return r

    def _z_score(self, x: Tensor, dim: int = -1) -> Tensor:
        """Z-score normalize along dim. Returns zeros if std < eps."""
        mean = x.mean(dim=dim, keepdim=True)
        std = x.std(dim=dim, keepdim=True)
        return (x - mean) / (std + self.eps)

    def _mix_logspace(self, log_p: Tensor, s_dense: Tensor, a: float) -> Tensor:
        """Blend log_p and s_dense in log-space, preserving log_p's scale.

        Algorithm:
            norm_log_p = z_score(log_p)          # unit variance
            norm_s     = z_score(s_dense)        # unit variance
            mixed_z    = (1-a) * norm_log_p + a * norm_s
            mixed      = mixed_z * std(log_p)    # restore log-prob scale

        The std-rescale step is load-bearing: without it, downstream
        softmax(mixed) over unit-variance scores produces a near-uniform
        distribution regardless of alpha, silently erasing the dense signal.
        """
        norm_log_p = self._z_score(log_p, dim=-1)
        norm_s = self._z_score(s_dense, dim=-1)
        mixed_z = (1.0 - a) * norm_log_p + a * norm_s
        log_p_std = log_p.std(dim=-1, keepdim=True)
        return mixed_z * log_p_std


class AlphaParams(nn.Module):
    """Learnable per-level alpha schedule. alpha_l = sigmoid(phi_l)."""

    def __init__(self, n_levels: int):
        super().__init__()
        self.phi = nn.Parameter(torch.zeros(n_levels))

    @property
    def alpha(self) -> list[float]:
        return torch.sigmoid(self.phi).tolist()
