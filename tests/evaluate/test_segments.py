"""Tests for per-segment labelling (audit bug B5)."""
import numpy as np
import pandas as pd

from evaluate.segments import (
    compute_item_popularity,
    label_cold_start_items,
    label_item_popularity_quartile,
    label_per_user_frame,
    label_user_history_quartile,
)


def test_compute_item_popularity_counts_every_occurrence():
    train_items = [1, 1, 2, 3, 3, 3, 4, 4]
    pop = compute_item_popularity(train_items)
    assert pop == {1: 2, 2: 1, 3: 3, 4: 2}


def test_cold_start_threshold_is_strict_less_than():
    pop = {1: 2, 2: 5, 3: 4, 4: 6}
    cold = label_cold_start_items(pop, threshold=5)
    # 5 is NOT cold; strictly less than 5 is cold.
    assert cold == {1, 3}


def test_history_quartile_labels_are_balanced():
    # 12 users with strictly increasing history length → quartiles balanced
    lengths = {uid: uid + 1 for uid in range(12)}
    q = label_user_history_quartile(lengths)

    assert set(q.values()) == {1, 2, 3, 4}
    counts = pd.Series(list(q.values())).value_counts().to_dict()
    assert counts == {1: 3, 2: 3, 3: 3, 4: 3}

    # Monotonically increasing input → quartile label monotonically
    # increasing. Q1 must be the shortest histories.
    q1_users = [u for u, v in q.items() if v == 1]
    q4_users = [u for u, v in q.items() if v == 4]
    assert max(lengths[u] for u in q1_users) < min(lengths[u] for u in q4_users)


def test_item_quartile_handles_ties_without_raising():
    # All same popularity — qcut on ranks still yields balanced quartiles
    # because `rank(method='first')` breaks ties by position.
    pop = {i: 3 for i in range(8)}
    q = label_item_popularity_quartile(pop)
    counts = pd.Series(list(q.values())).value_counts().to_dict()
    assert set(counts.values()) == {2}


def test_label_per_user_frame_joins_segments():
    per_user = pd.DataFrame({
        "job_name": ["j"] * 4,
        "dataset": ["beauty"] * 4,
        "decoder_type": ["mtl"] * 4,
        "decoding_strategy": ["level_aware_mix_learned"] * 4,
        "alpha_schedule": ["[]"] * 4,
        "seed": [42] * 4,
        "n_cands": [200] * 4,
        "beam_size": [50] * 4,
        "user_id": ["0", "1", "2", "3"],
        "metric": ["recall@10"] * 4,
        "value": [0.0, 1.0, 0.0, 1.0],
    })

    user_q = {0: 1, 1: 2, 2: 3, 3: 4}
    item_q = {100: 1, 101: 4}
    cold = {100}
    target = {0: 100, 1: 101, 2: 999, 3: 100}  # 999 unknown → NaN

    labelled = label_per_user_frame(
        per_user,
        user_history_quartile=user_q,
        item_popularity_quartile=item_q,
        cold_start_items=cold,
        target_item_by_user=target,
    )

    # Original rows preserved
    assert len(labelled) == 4
    # user_quartile joined
    assert labelled.loc[labelled["user_id"] == "0", "user_quartile"].iloc[0] == 1.0
    # item_quartile joined and NaN for unknown target
    assert labelled.loc[labelled["user_id"] == "1", "item_quartile"].iloc[0] == 4.0
    assert np.isnan(labelled.loc[labelled["user_id"] == "2", "item_quartile"].iloc[0])
    # cold_start flag
    cold_col = labelled.set_index("user_id")["cold_start"].to_dict()
    assert cold_col["0"] is True or cold_col["0"] == True  # noqa: E712
    assert cold_col["1"] is False or cold_col["1"] == False  # noqa: E712


def test_label_per_user_frame_without_target_items_is_safe():
    per_user = pd.DataFrame({
        "job_name": ["j"],
        "dataset": ["beauty"],
        "decoder_type": ["vanilla"],
        "decoding_strategy": ["vanilla"],
        "alpha_schedule": ["[]"],
        "seed": [0],
        "n_cands": [200],
        "beam_size": [50],
        "user_id": ["0"],
        "metric": ["recall@10"],
        "value": [1.0],
    })
    labelled = label_per_user_frame(per_user)
    assert np.isnan(labelled.iloc[0]["user_quartile"])
    assert np.isnan(labelled.iloc[0]["item_quartile"])
    assert labelled.iloc[0]["cold_start"] == False  # noqa: E712
