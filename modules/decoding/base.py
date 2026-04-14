from abc import ABC, abstractmethod
from collections.abc import Callable

from torch import Tensor


class BeamSearchStrategy(ABC):
    @abstractmethod
    def expand(
        self,
        beams: Tensor | None,        # [B, k, h] or None (first level)
        log_probas: Tensor,             # [B, k] cumulative log-probs (zeros at h=0)
        probas: Tensor,                 # [B*k, vocab] or [B, vocab] softmax output
        h: int,                         # current codebook level (0-indexed)
        n_cands: int,                   # candidates to sample/take
        check_valid_fn: Callable,       # bound _check_valid_prefix — MUST be called
        codebook_embs: list[Tensor],    # List[L] of [vocab, 32] codebook embedding tables
        decoder_hidden: Tensor | None = None,  # [B*k, d_model] for hybrid strategies
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Returns: (new_beams [B, k, h+1], new_log_probas [B, k], parent_global_idx [B*k])"""
        ...
