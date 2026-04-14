"""Collect all eval result JSONs from S3 and assemble results/all_runs.parquet.

Usage (from compute machine):
    python scripts/collect_results.py --bucket YOUR_S3_BUCKET --prefix rqvae-level-aware
"""
import argparse
import json
import os
from pathlib import Path

import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def download_results(bucket: str, prefix: str, local_dir: str) -> list[dict]:
    """Download all result JSONs from S3 and return as list of dicts."""
    s3 = boto3.client("s3")
    results = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=f"{prefix}/results/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".json"):
                continue
            local_path = Path(local_dir) / Path(key).name
            s3.download_file(bucket, key, str(local_path))
            with open(local_path) as f:
                results.append(json.load(f))
    return results


def results_to_dataframe(results: list[dict]) -> pd.DataFrame:
    """Flatten result dicts into a DataFrame. per_user stored as JSON string."""
    rows = []
    for r in results:
        row = {
            "job_name": r.get("job_name", ""),
            "dataset": r["dataset"],
            "decoder_type": r["decoder_type"],
            "decoding_strategy": r["decoding_strategy"],
            "alpha_schedule": json.dumps(r.get("alpha_schedule", [])),
            "seed": r.get("seed", -1),
            "n_cands": r.get("n_cands", 200),
            "beam_size": r.get("beam_size", 50),
        }
        # Flatten aggregate metrics
        for k, v in r.get("aggregate", {}).items():
            row[k] = v
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", default="YOUR_S3_BUCKET")
    parser.add_argument("--prefix", default="rqvae-level-aware")
    parser.add_argument("--local-results", default="results/raw_json")
    parser.add_argument("--output", default="results/all_runs.parquet")
    args = parser.parse_args()

    os.makedirs(args.local_results, exist_ok=True)

    print(f"Downloading from s3://{args.bucket}/{args.prefix}/...")
    results = download_results(args.bucket, args.prefix, args.local_results)
    print(f"Downloaded {len(results)} result files.")

    df = results_to_dataframe(results)
    pq.write_table(pa.Table.from_pandas(df), args.output)
    print(f"Saved {len(df)} rows to {args.output}")
    print(df.groupby(["dataset", "decoder_type", "decoding_strategy"]).size())


if __name__ == "__main__":
    main()
