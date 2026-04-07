"""Tests for evaluate/metrics.py — TopKAccumulator with NDCG and per-user tracking."""
import math
import torch
import pytest

from evaluate.metrics import TopKAccumulator


def _make_accumulator(ks=(1, 5, 10, 20)):
    return TopKAccumulator(ks=list(ks))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _target_tensor(sid_tuple):
    """Return [1, L] target tensor from a tuple of ints."""
    return torch.tensor([list(sid_tuple)], dtype=torch.long)


def _beam_tensor(beam_list):
    """Return [1, k, L] beam tensor from a list of tuples."""
    return torch.tensor([[list(t) for t in beam_list]], dtype=torch.long)


# ---------------------------------------------------------------------------
# test_recall_at_k_hit
# ---------------------------------------------------------------------------

def test_recall_at_k_hit():
    """Target is rank-1 beam → recall@1, @5, @10 all 1.0."""
    target = [1, 2, 3]
    beams = [
        [1, 2, 3],  # rank 1 — hit
        [4, 5, 6],
        [7, 8, 9],
        [1, 2, 4],
        [0, 0, 0],
    ]
    acc = _make_accumulator(ks=(1, 5, 10))
    acc.accumulate(
        generated_ids=_beam_tensor(beams),
        target_ids=_target_tensor(target),
        user_ids=[0],
    )
    result = acc.reduce()
    assert result["recall@1"] == pytest.approx(1.0)
    assert result["recall@5"] == pytest.approx(1.0)
    assert result["recall@10"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# test_recall_at_k_miss
# ---------------------------------------------------------------------------

def test_recall_at_k_miss():
    """Target not in top-5 but in top-10 → recall@5=0, recall@10=1."""
    target = [9, 9, 9]
    # Beams indexed 0-9; target at position 7 (0-indexed) → rank 8 (1-indexed)
    beams = [
        [0, 0, 0],  # 0
        [1, 1, 1],  # 1
        [2, 2, 2],  # 2
        [3, 3, 3],  # 3
        [4, 4, 4],  # 4
        [5, 5, 5],  # 5
        [6, 6, 6],  # 6
        [9, 9, 9],  # 7 ← target (0-indexed rank 7 → < 10 but >= 5)
        [8, 8, 8],  # 8
        [7, 7, 7],  # 9
    ]
    acc = _make_accumulator(ks=(5, 10))
    acc.accumulate(
        generated_ids=_beam_tensor(beams),
        target_ids=_target_tensor(target),
        user_ids=[42],
    )
    result = acc.reduce()
    assert result["recall@5"] == pytest.approx(0.0)
    assert result["recall@10"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# test_ndcg_at_k
# ---------------------------------------------------------------------------

def test_ndcg_at_k():
    """Target at rank 3 (1-indexed) → ndcg@5 = 1/log2(4), ndcg@10 same, ndcg@1=0."""
    target = [3, 3, 3]
    beams = [
        [0, 0, 0],  # rank 1
        [1, 1, 1],  # rank 2
        [3, 3, 3],  # rank 3 ← target
        [4, 4, 4],  # rank 4
        [5, 5, 5],  # rank 5
        [6, 6, 6],
        [7, 7, 7],
        [8, 8, 8],
        [9, 9, 9],
        [2, 2, 2],
    ]
    acc = _make_accumulator(ks=(1, 5, 10))
    acc.accumulate(
        generated_ids=_beam_tensor(beams),
        target_ids=_target_tensor(target),
        user_ids=[0],
    )
    result = acc.reduce()
    expected_ndcg = 1.0 / math.log2(3 + 1)  # rank=3, denom = log2(4)
    assert result["ndcg@1"] == pytest.approx(0.0)
    assert result["ndcg@5"] == pytest.approx(expected_ndcg)
    assert result["ndcg@10"] == pytest.approx(expected_ndcg)


# ---------------------------------------------------------------------------
# test_ndcg_at_rank1
# ---------------------------------------------------------------------------

def test_ndcg_at_rank1():
    """Target at rank 1 → ndcg@1 = 1.0."""
    target = [1, 2, 3]
    beams = [
        [1, 2, 3],  # rank 1 ← target
        [4, 5, 6],
        [7, 8, 9],
    ]
    acc = _make_accumulator(ks=(1, 5))
    acc.accumulate(
        generated_ids=_beam_tensor(beams),
        target_ids=_target_tensor(target),
        user_ids=[0],
    )
    result = acc.reduce()
    assert result["ndcg@1"] == pytest.approx(1.0)
    assert result["ndcg@5"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# test_per_user_tracking
# ---------------------------------------------------------------------------

def test_per_user_tracking():
    """Batch of 3 users; verify per_user dict has correct keys and values."""
    # User 0: hit at rank 1
    # User 1: hit at rank 6 (0-indexed 5)
    # User 2: miss entirely
    ks = (1, 5, 10)
    acc = _make_accumulator(ks=ks)

    # Build batch tensors: [3, k=10, L=2]
    target = torch.tensor(
        [
            [1, 1],   # user 0
            [6, 6],   # user 1
            [99, 99], # user 2
        ],
        dtype=torch.long,
    )
    beams = torch.tensor(
        [
            # user 0: target [1,1] at beam index 0
            [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9], [10, 10]],
            # user 1: target [6,6] at beam index 5
            [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9], [10, 10]],
            # user 2: target [99,99] not in beams
            [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9], [10, 10]],
        ],
        dtype=torch.long,
    )

    acc.accumulate(
        generated_ids=beams,
        target_ids=target,
        user_ids=[100, 101, 102],
    )
    result = acc.reduce()
    per_user = result["per_user"]

    assert set(per_user.keys()) == {100, 101, 102}

    expected_keys = {f"recall@{k}" for k in ks} | {f"ndcg@{k}" for k in ks}
    for uid in (100, 101, 102):
        assert set(per_user[uid].keys()) == expected_keys

    # User 100: hit at rank 1 (0-indexed 0)
    assert per_user[100]["recall@1"] == pytest.approx(1.0)
    assert per_user[100]["ndcg@1"] == pytest.approx(1.0)

    # User 101: hit at 0-indexed rank 5, so 1-indexed rank 6
    assert per_user[101]["recall@5"] == pytest.approx(0.0)  # rank 6 > 5
    assert per_user[101]["recall@10"] == pytest.approx(1.0)
    expected_ndcg_6 = 1.0 / math.log2(6 + 1)
    assert per_user[101]["ndcg@10"] == pytest.approx(expected_ndcg_6)
    assert per_user[101]["ndcg@5"] == pytest.approx(0.0)

    # User 102: complete miss
    for k in ks:
        assert per_user[102][f"recall@{k}"] == pytest.approx(0.0)
        assert per_user[102][f"ndcg@{k}"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# test_no_user_ids
# ---------------------------------------------------------------------------

def test_no_user_ids():
    """When user_ids=None, auto-assigned sequential IDs still work."""
    target = _target_tensor([1, 2, 3])
    beams = _beam_tensor([[1, 2, 3], [4, 5, 6]])

    acc = _make_accumulator(ks=(1, 5))
    acc.accumulate(generated_ids=beams, target_ids=target)  # no user_ids arg
    result = acc.reduce()

    assert "per_user" in result
    # Auto-assigned ID 0 should be present
    assert 0 in result["per_user"]
    assert result["recall@1"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# test_backward_compat
# ---------------------------------------------------------------------------

def test_backward_compat():
    """reduce() still returns recall@k keys for code that doesn't use ndcg."""
    acc = _make_accumulator(ks=(5, 10, 20))
    target = _target_tensor([1, 1, 1])
    beams = _beam_tensor([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
    acc.accumulate(generated_ids=beams, target_ids=target)
    result = acc.reduce()
    # Must have recall@k keys
    assert "recall@5" in result
    assert "recall@10" in result
    assert "recall@20" in result
    # Must also have ndcg@k keys
    assert "ndcg@5" in result
    assert "ndcg@10" in result
    assert "ndcg@20" in result
    # per_user must exist
    assert "per_user" in result
