import json
import pathlib
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

RESULT_SCHEMA = {
    "job_name": str,
    "dataset": str,           # beauty|sports|toys|steam
    "decoder_type": str,      # vanilla|mtl|liger
    "decoding_strategy": str,
    "alpha_schedule": list,   # [float, ...] or []
    "seed": int,
    "aggregate": dict,        # {"recall@5": float, ...}
    # per_user stored separately for space
}


def write_result(result: dict, output_path: str) -> None:
    """Write a single result dict as JSON.

    Creates parent directories as needed.
    """
    path = pathlib.Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)


def load_all_results(results_dir: str) -> pd.DataFrame:
    """Load all JSON result files from results_dir into a DataFrame.

    Each JSON file corresponds to one row. Nested dicts (e.g. ``aggregate``)
    are flattened one level: keys become ``<parent>__<child>`` columns.
    The ``per_user`` key (if present) is dropped to keep the DataFrame flat.
    """
    results_path = pathlib.Path(results_dir)
    rows: list[dict[str, Any]] = []

    for json_file in sorted(results_path.glob("**/*.json")):
        with json_file.open("r", encoding="utf-8") as fh:
            data: dict[str, Any] = json.load(fh)

        # Remove per_user — too large and not needed for aggregate analysis
        data.pop("per_user", None)

        flat: dict[str, Any] = {}
        for key, value in data.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    flat[f"{key}__{sub_key}"] = sub_value
            else:
                flat[key] = value

        flat["_source_file"] = str(json_file)
        rows.append(flat)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def save_parquet(df: pd.DataFrame, path: str) -> None:
    """Save a DataFrame to a Parquet file."""
    out_path = pathlib.Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, str(out_path))


def load_parquet(path: str) -> pd.DataFrame:
    """Load a Parquet file into a DataFrame."""
    table = pq.read_table(str(path))
    return table.to_pandas()
