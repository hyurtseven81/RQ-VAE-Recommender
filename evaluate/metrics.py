import math
from collections import defaultdict

from torch import Tensor


class TopKAccumulator:
    def __init__(self, ks=[1, 5, 10, 20]):
        self.ks = ks
        self.reset()

    def reset(self):
        self.total = 0
        self.metrics = defaultdict(int)
        self.ndcg_sums = defaultdict(float)
        self.per_user: dict = {}

    def accumulate(
        self,
        generated_ids: Tensor,
        target_ids: Tensor,
        user_ids: list[int] | None = None,
    ) -> None:
        """Accumulate metrics for a batch.

        Args:
            generated_ids: Beam outputs of shape [B, k, L] where B is batch
                size, k is beam width, L is number of hierarchy levels.
            target_ids: Ground-truth SID tuples of shape [B, L].
            user_ids: Optional list of user IDs of length B. When None,
                sequential integers starting from self.total are used.
        """
        B = target_ids.shape[0]

        if user_ids is None:
            user_ids = list(range(self.total, self.total + B))

        # generated_ids: [B, k, L], target_ids: [B, L]
        # For each sample in the batch find if/where the target appears in
        # the beam list.
        # pos_match[b, beam] is True when all L levels of beam match target.
        pos_match = (
            target_ids.unsqueeze(1) == generated_ids
        ).all(dim=-1)  # [B, k]

        # max along beam dim: match_found[b] = True if any beam matches;
        # rank[b] = index of first True (or 0 when none matched — we use
        # match_found to discriminate).
        match_found, rank = pos_match.max(dim=-1)  # [B], [B]

        for b_idx in range(B):
            uid = user_ids[b_idx]
            user_entry: dict = {}

            for k in self.ks:
                hit = bool(match_found[b_idx]) and int(rank[b_idx]) < k
                self.metrics[f"recall@{k}"] += int(hit)

                # For backward compat keep old h@ keys if ks == default set
                # that includes 1/5/10 (only when they were requested)
                user_entry[f"recall@{k}"] = float(hit)

            # NDCG — 1-indexed rank
            if bool(match_found[b_idx]):
                r_one_indexed = int(rank[b_idx]) + 1  # convert 0-indexed → 1-indexed
            else:
                r_one_indexed = None  # no match

            for k in self.ks:
                if r_one_indexed is not None and r_one_indexed <= k:
                    ndcg_val = 1.0 / math.log2(r_one_indexed + 1)
                else:
                    ndcg_val = 0.0
                self.ndcg_sums[f"ndcg@{k}"] += ndcg_val
                user_entry[f"ndcg@{k}"] = ndcg_val

            self.per_user[uid] = user_entry

        self.total += B

    def reduce(self) -> dict:
        """Return aggregate metrics plus the per_user dict.

        Backward-compatible: the recall@K keys are always present.
        """
        result: dict = {}
        for k, v in self.metrics.items():
            result[k] = v / self.total if self.total > 0 else 0.0
        for k, v in self.ndcg_sums.items():
            result[k] = v / self.total if self.total > 0 else 0.0
        result["per_user"] = self.per_user
        return result
