"""Tests for evaluate.result_store (JSON/Parquet I/O for eval runs)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("pyarrow")  # parquet tests are skipped without pyarrow

from evaluate.result_store import (
    load_all_results,
    load_parquet,
    save_parquet,
    write_result,
)


def _sample(seed: int) -> dict:
    return {
        "job_name": f"j{seed}",
        "dataset": "beauty",
        "decoder_type": "mtl",
        "decoding_strategy": "level_aware_mix_learned",
        "alpha_schedule": [0.6, 0.3, 0.1],
        "seed": seed,
        "aggregate": {"recall@10": 0.12 + 0.01 * seed, "ndcg@10": 0.08},
        "per_user": {"0": {"recall@10": 1.0}},
    }


def test_write_result_creates_parent_dirs(tmp_path: Path):
    out = tmp_path / "nested" / "dir" / "r.json"
    write_result(_sample(1), str(out))
    assert out.exists()
    loaded = json.loads(out.read_text())
    assert loaded["seed"] == 1
    assert loaded["aggregate"]["recall@10"] == pytest.approx(0.13)


def test_load_all_results_flattens_one_level_and_drops_per_user(tmp_path: Path):
    write_result(_sample(1), str(tmp_path / "a.json"))
    write_result(_sample(2), str(tmp_path / "nested" / "b.json"))

    df = load_all_results(str(tmp_path))

    # 2 files -> 2 rows, sorted by path so j1 comes first.
    assert len(df) == 2
    # Nested aggregate dict got flattened with "__" separator.
    assert "aggregate__recall@10" in df.columns
    assert "aggregate__ndcg@10" in df.columns
    # per_user dropped — too large for aggregate analysis.
    assert not any(col.startswith("per_user") for col in df.columns)
    # Non-dict fields kept as-is.
    assert df["job_name"].tolist() == ["j1", "j2"]
    # Source-file trace attached.
    assert "_source_file" in df.columns


def test_load_all_results_empty_dir_returns_empty_frame(tmp_path: Path):
    df = load_all_results(str(tmp_path))
    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_parquet_roundtrip(tmp_path: Path):
    df_in = pd.DataFrame({
        "job_name": ["j1", "j2"],
        "dataset": ["beauty", "sports"],
        "recall@10": [0.12, 0.09],
    })
    path = tmp_path / "out.parquet"
    save_parquet(df_in, str(path))
    assert path.exists()

    df_out = load_parquet(str(path))
    pd.testing.assert_frame_equal(
        df_in.reset_index(drop=True),
        df_out.reset_index(drop=True),
        check_like=True,
    )
