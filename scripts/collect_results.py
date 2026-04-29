"""Collect all eval result JSONs from S3 and assemble results parquets.

Produces two parquets (fixes audit bug B4 — prior version silently dropped
per-user metrics, making paired bootstrap and per-segment analysis impossible):

- ``results/all_runs.parquet``       one row per (job, dataset, strategy, seed)
                                     with the aggregate metrics flattened into
                                     columns. Consumed by make_tables.py and
                                     make_figures.py.
- ``results/per_user_runs.parquet``  long-form one row per (job, user_id,
                                     metric) — needed for paired bootstrap
                                     and per-segment slicing. Consumed by
                                     evaluate.stats and the segment-aware
                                     table generation.

Usage (from the compute machine)::

    # --bucket defaults to the bucket in $RQVAE_S3_BASE
    python scripts/collect_results.py --prefix rqvae-level-aware
    python scripts/collect_results.py --bucket <your-bucket> --prefix rqvae-level-aware
"""
import argparse
import json
import os
from pathlib import Path

import pandas as pd

# boto3 / pyarrow are heavy deps only needed inside main(). Lazy-importing
# them keeps the pure DataFrame helpers testable in environments without
# the full SageMaker stack installed.


def download_results(bucket: str, prefix: str, local_dir: str) -> list[dict]:
    """Download all result JSONs from S3 and return as list of dicts.

    run_eval.py writes result JSONs to /opt/ml/output/data which SageMaker
    packs into an ``output.tar.gz``. This function accepts a prefix that
    points at (or above) the eval-results tree, downloads every object
    under it ending in ``.json`` or ``.tar.gz``, extracts tarballs, and
    returns the union of parsed JSONs.

    AWS credentials come from the usual boto3 chain; ``RQVAE_AWS_PROFILE``
    (or ``AWS_PROFILE``) selects the profile used for the S3 scan.
    """
    import tarfile

    import boto3  # lazy: not needed for helper tests

    profile = os.environ.get("RQVAE_AWS_PROFILE") or os.environ.get("AWS_PROFILE")
    region = (
        os.environ.get("RQVAE_AWS_REGION")
        or os.environ.get("AWS_REGION")
        or "us-east-1"
    )
    boto_sess = boto3.Session(profile_name=profile, region_name=region)
    s3 = boto_sess.client("s3")

    results: list[dict] = []
    paginator = s3.get_paginator("list_objects_v2")
    # Previously this function hardcoded an extra ``/results/`` suffix which
    # never existed on S3. Paginate under the caller-supplied prefix directly.
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix.rstrip("/") + "/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            fname = Path(key).name
            if not (key.endswith(".json") or key.endswith(".tar.gz")):
                continue
            local_path = Path(local_dir) / key.replace("/", "__")
            local_path.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket, key, str(local_path))

            if fname.endswith(".json"):
                try:
                    with local_path.open() as f:
                        results.append(json.load(f))
                except (json.JSONDecodeError, OSError) as e:
                    print(f"[warn] could not parse {key}: {e}")
                continue

            # tar.gz: extract and sweep for any *.json file inside
            extract_dir = local_path.with_suffix("").with_suffix(".extracted")
            extract_dir.mkdir(parents=True, exist_ok=True)
            try:
                with tarfile.open(local_path) as tf:
                    tf.extractall(extract_dir)
            except (tarfile.TarError, OSError) as e:
                print(f"[warn] could not extract {key}: {e}")
                continue
            for json_file in extract_dir.rglob("*.json"):
                try:
                    with json_file.open() as f:
                        results.append(json.load(f))
                except (json.JSONDecodeError, OSError) as e:
                    print(f"[warn] could not parse {json_file}: {e}")

    return results


def _run_identity(r: dict) -> dict:
    """Common identity columns shared by aggregate and per-user frames."""
    return {
        "job_name": r.get("job_name", ""),
        "dataset": r["dataset"],
        "decoder_type": r["decoder_type"],
        "decoding_strategy": r["decoding_strategy"],
        "alpha_schedule": json.dumps(r.get("alpha_schedule", [])),
        "seed": r.get("seed", -1),
        "n_cands": r.get("n_cands", 200),
        "beam_size": r.get("beam_size", 50),
    }


def aggregate_to_dataframe(results: list[dict]) -> pd.DataFrame:
    """One row per run with aggregate metrics flattened."""
    rows = []
    for r in results:
        row = _run_identity(r)
        for k, v in r.get("aggregate", {}).items():
            row[k] = v
        rows.append(row)
    return pd.DataFrame(rows)


def per_user_to_dataframe(results: list[dict]) -> pd.DataFrame:
    """Long-form frame: one row per (run, user_id, metric).

    Schema: identity cols + user_id (str) + metric (str) + value (float).

    We keep metric long-form rather than a wide user_metric column so that
    downstream code can group-by metric and segment label without caring
    which metrics the run emitted.
    """
    rows = []
    for r in results:
        identity = _run_identity(r)
        per_user = r.get("per_user", {}) or {}
        for user_id, metrics in per_user.items():
            if not isinstance(metrics, dict):
                continue
            for metric_name, value in metrics.items():
                row = dict(identity)
                row["user_id"] = str(user_id)
                row["metric"] = metric_name
                row["value"] = float(value)
                rows.append(row)
    return pd.DataFrame(rows)


def main():
    import pyarrow as pa  # lazy: not needed for helper tests
    import pyarrow.parquet as pq

    # Default bucket is the one encoded in RQVAE_S3_BASE (shape s3://bucket/prefix).
    default_bucket = None
    env_base = os.environ.get("RQVAE_S3_BASE", "").removeprefix("s3://").strip("/")
    if env_base:
        default_bucket = env_base.split("/", 1)[0]
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", default=default_bucket, required=default_bucket is None)
    parser.add_argument("--prefix", default="rqvae-level-aware")
    parser.add_argument("--local-results", default="results/raw_json")
    parser.add_argument(
        "--aggregate-output",
        default="results/all_runs.parquet",
        help="Path for one-row-per-run aggregate parquet.",
    )
    parser.add_argument(
        "--per-user-output",
        default="results/per_user_runs.parquet",
        help="Path for long-form per-user parquet (empty file if no per-user data).",
    )
    args = parser.parse_args()

    os.makedirs(args.local_results, exist_ok=True)
    for out_path in (args.aggregate_output, args.per_user_output):
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    print(f"Downloading from s3://{args.bucket}/{args.prefix}/...")
    results = download_results(args.bucket, args.prefix, args.local_results)
    print(f"Downloaded {len(results)} result files.")

    agg_df = aggregate_to_dataframe(results)
    pq.write_table(pa.Table.from_pandas(agg_df), args.aggregate_output)
    print(f"Saved {len(agg_df)} aggregate rows to {args.aggregate_output}")

    pu_df = per_user_to_dataframe(results)
    pq.write_table(pa.Table.from_pandas(pu_df), args.per_user_output)
    print(f"Saved {len(pu_df)} per-user rows to {args.per_user_output}")

    if not agg_df.empty:
        print(agg_df.groupby(["dataset", "decoder_type", "decoding_strategy"]).size())


if __name__ == "__main__":
    main()
