"""User / item segmentation for per-segment eval analysis.

Paper Table T3 slices results by:
  * user history quartile (Q1=sparse, Q4=dense),
  * item popularity quartile,
  * cold-start items (train occurrences below a threshold).

These labels are computed once per dataset from the training split and then
joined into the per-user parquet produced by scripts/collect_results.py.

Keeping this logic in a single module (rather than inlining it in
run_eval.py or metrics.py) means offline analysis and the eval loop share
exactly the same segmentation — important when reporting significance on
per-segment differences.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd


def compute_item_popularity(train_item_ids: Iterable[int]) -> dict[int, int]:
    """Count train occurrences per item id.

    Args:
        train_item_ids: flat iterable of item ids from the training split
            (target items, not sequence history). Expect one entry per
            observed interaction.

    Returns:
        ``{item_id: occurrence_count}``. Items never seen in training are
        absent; callers should default them to 0.
    """
    return dict(Counter(int(i) for i in train_item_ids))


def label_cold_start_items(
    popularity: dict[int, int],
    threshold: int = 5,
) -> set[int]:
    """Return the set of item ids with train occurrences strictly below
    ``threshold`` (default follows plan §9.5: cold-start = seen <5 times).
    """
    return {item_id for item_id, count in popularity.items() if count < threshold}


def _quartile_labels(values: Sequence[float]) -> np.ndarray:
    """Return a 1-indexed quartile label array (1=lowest, 4=highest).

    Ties are broken by rank so quartiles have roughly equal mass. Returns
    int8 array of the same length as ``values``.
    """
    if len(values) == 0:
        return np.empty(0, dtype=np.int8)
    ranks = pd.Series(values).rank(method="first", ascending=True).to_numpy()
    # qcut on ranks guarantees equal-sized buckets even with many duplicates.
    quartile = pd.qcut(ranks, q=4, labels=[1, 2, 3, 4])
    return np.asarray(quartile, dtype=np.int8)


def label_user_history_quartile(
    user_history_lengths: dict[int, int],
) -> dict[int, int]:
    """Bucket users into history-length quartiles.

    Q1 = sparse (shortest histories), Q4 = dense (longest histories).
    Returns ``{user_id: quartile}`` with ``quartile ∈ {1, 2, 3, 4}``.
    """
    if not user_history_lengths:
        return {}
    user_ids = list(user_history_lengths.keys())
    lengths = [user_history_lengths[u] for u in user_ids]
    labels = _quartile_labels(lengths)
    return {uid: int(q) for uid, q in zip(user_ids, labels)}


def label_item_popularity_quartile(
    popularity: dict[int, int],
) -> dict[int, int]:
    """Bucket items into popularity quartiles.

    Q1 = tail (least popular), Q4 = head. Items absent from ``popularity``
    are not labelled — callers should treat them as cold-start separately.
    """
    if not popularity:
        return {}
    item_ids = list(popularity.keys())
    counts = [popularity[i] for i in item_ids]
    labels = _quartile_labels(counts)
    return {iid: int(q) for iid, q in zip(item_ids, labels)}


def label_per_user_frame(
    per_user: pd.DataFrame,
    user_history_quartile: dict[int, int] | None = None,
    item_popularity_quartile: dict[int, int] | None = None,
    cold_start_items: set[int] | None = None,
    target_item_by_user: dict[int, int] | None = None,
) -> pd.DataFrame:
    """Add ``user_quartile``, ``item_quartile``, ``cold_start`` columns to
    the long-form per-user frame produced by
    :func:`scripts.collect_results.per_user_to_dataframe`.

    Join keys:
      * user_history_quartile → joined on ``user_id``.
      * item_popularity_quartile / cold_start → joined on the target item
        id for that user, supplied via ``target_item_by_user``. When
        absent those columns are all NaN/False.

    The frame is returned with new columns appended; the original rows are
    preserved unchanged. Missing users/items produce NaN (or False for the
    boolean column).
    """
    out = per_user.copy()

    if user_history_quartile:

        def _user_q(uid: str) -> float:
            try:
                return float(user_history_quartile[int(uid)])
            except (KeyError, ValueError):
                return float("nan")

        out["user_quartile"] = out["user_id"].map(_user_q)
    else:
        out["user_quartile"] = float("nan")

    if target_item_by_user is not None:

        def _target_item(uid: str) -> float:
            try:
                return float(target_item_by_user[int(uid)])
            except (KeyError, ValueError):
                return float("nan")

        out["target_item"] = out["user_id"].map(_target_item)

        def _item_q(iid: float) -> float:
            if item_popularity_quartile is None or np.isnan(iid):
                return float("nan")
            return float(item_popularity_quartile.get(int(iid), float("nan")))

        def _cold(iid: float) -> bool:
            if cold_start_items is None or np.isnan(iid):
                return False
            return int(iid) in cold_start_items

        out["item_quartile"] = out["target_item"].map(_item_q)
        out["cold_start"] = out["target_item"].map(_cold)
    else:
        out["item_quartile"] = float("nan")
        out["cold_start"] = False

    return out
