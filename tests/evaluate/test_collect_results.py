"""Tests for scripts/collect_results.py (audit bug B4 regression coverage).

Verifies that per_user data is preserved into a long-form parquet alongside
the aggregate frame. Pre-fix, per_user was silently dropped, which broke
paired-bootstrap and per-segment slicing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.collect_results import aggregate_to_dataframe, per_user_to_dataframe  # noqa: E402


def _sample_result(
    job: str,
    dataset: str = "beauty",
    decoder: str = "mtl",
    strategy: str = "level_aware_mix_learned",
    seed: int = 42,
    per_user: dict | None = None,
) -> dict:
    return {
        "job_name": job,
        "dataset": dataset,
        "decoder_type": decoder,
        "decoding_strategy": strategy,
        "alpha_schedule": [0.6, 0.3, 0.1],
        "seed": seed,
        "n_cands": 200,
        "beam_size": 50,
        "aggregate": {"recall@10": 0.12, "ndcg@10": 0.08},
        "per_user": per_user
        if per_user is not None
        else {
            "0": {"recall@10": 1.0, "ndcg@10": 1.0},
            "1": {"recall@10": 0.0, "ndcg@10": 0.0},
        },
    }


def test_aggregate_flattens_metrics():
    results = [_sample_result("j1"), _sample_result("j2", seed=7)]
    df = aggregate_to_dataframe(results)

    assert len(df) == 2
    assert set(df["job_name"]) == {"j1", "j2"}
    assert "recall@10" in df.columns and "ndcg@10" in df.columns
    # alpha_schedule round-trips through JSON serialization.
    assert df["alpha_schedule"].iloc[0] == "[0.6, 0.3, 0.1]"


def test_per_user_long_form_has_one_row_per_user_metric_pair():
    results = [_sample_result("j1")]
    df = per_user_to_dataframe(results)

    # 2 users × 2 metrics = 4 rows.
    assert len(df) == 4
    assert set(df["user_id"]) == {"0", "1"}
    assert set(df["metric"]) == {"recall@10", "ndcg@10"}
    assert df["value"].dtype == float

    # Identity columns replicated onto every per-user row.
    assert (df["job_name"] == "j1").all()
    assert (df["dataset"] == "beauty").all()


def test_per_user_missing_block_returns_empty_frame():
    # Pre-fix behavior silently lost per_user. Post-fix: if a run genuinely
    # has no per-user data, we return an empty frame rather than crashing
    # or fabricating rows.
    result_without_per_user = _sample_result("j1")
    result_without_per_user.pop("per_user")
    df = per_user_to_dataframe([result_without_per_user])
    assert df.empty


def test_per_user_row_count_matches_users_times_metrics():
    results = [
        _sample_result(
            "j1",
            per_user={
                f"u{i}": {"recall@5": float(i % 2), "ndcg@5": float(i % 2) * 0.5}
                for i in range(10)
            },
        )
    ]
    df = per_user_to_dataframe(results)
    assert len(df) == 10 * 2

    # Spot-check a value.
    row = df[(df["user_id"] == "u3") & (df["metric"] == "ndcg@5")].iloc[0]
    assert row["value"] == pytest.approx(0.5)


def test_per_user_ignores_malformed_metric_block():
    # Defensive: if a run serialized per_user as something other than a
    # dict-of-dicts (e.g. a list, or None), we skip that user silently
    # rather than crashing the whole collection job.
    results = [
        _sample_result(
            "j1",
            per_user={
                "0": {"recall@10": 1.0},
                "1": None,            # malformed
                "2": [0.5, 0.5],      # malformed
                "3": {"recall@10": 0.0},
            },
        )
    ]
    df = per_user_to_dataframe(results)

    # Only the two well-formed users contribute rows.
    assert set(df["user_id"]) == {"0", "3"}
