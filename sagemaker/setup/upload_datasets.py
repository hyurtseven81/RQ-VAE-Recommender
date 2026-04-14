"""Run dataset preprocessing locally then upload processed data to S3.

Run this once from the compute machine before launching SageMaker training jobs.
All datasets are preprocessed and uploaded to:

    s3://REDACTED-BUCKET/rqvae-level-aware/datasets/<dataset>/

Usage::

    python sagemaker/setup/upload_datasets.py --datasets all
    python sagemaker/setup/upload_datasets.py --datasets steam
    python sagemaker/setup/upload_datasets.py --datasets beauty sports toys steam
"""
import argparse
import os
import sys

import boto3
import botocore


S3_BUCKET = "REDACTED-BUCKET"
S3_PREFIX = "rqvae-level-aware/datasets"
AWS_PROFILE = "REDACTED-PROFILE"

DATASET_LOCAL_DIRS = {
    "amazon": "dataset/amazon",
    "steam": "dataset/steam",
    "ml-1m": "dataset/ml-1m",
    "ml-32m": "dataset/ml-32m",
}

ALL_DATASETS = list(DATASET_LOCAL_DIRS.keys())


def preprocess_dataset(dataset: str, local_dir: str) -> None:
    """Trigger local preprocessing for the given dataset if not already done."""
    print(f"[preprocess] {dataset} -> {local_dir}")
    os.makedirs(local_dir, exist_ok=True)

    if dataset == "amazon":
        from data.amazon import AmazonReviews
        for split in ["beauty", "sports", "toys"]:
            print(f"  Processing Amazon split='{split}' …")
            raw = AmazonReviews(root=local_dir, split=split)
            if not os.path.exists(raw.processed_paths[0]):
                raw.process()

    elif dataset == "steam":
        from data.steam import RawSteam
        print("  Processing Steam …")
        raw = RawSteam(root=local_dir)
        if not os.path.exists(raw.processed_paths[0]):
            raw.process()

    elif dataset == "ml-1m":
        from data.ml1m import RawMovieLens1M
        print("  Processing MovieLens-1M …")
        raw = RawMovieLens1M(root=local_dir)
        if not os.path.exists(raw.processed_paths[0]):
            raw.process()

    elif dataset == "ml-32m":
        from data.ml32m import RawMovieLens32M
        print("  Processing MovieLens-32M …")
        raw = RawMovieLens32M(root=local_dir)
        if not os.path.exists(raw.processed_paths[0]):
            raw.process()

    else:
        print(f"  [WARN] Unknown dataset '{dataset}', skipping preprocessing.")


def upload_directory(s3_client, local_dir: str, bucket: str, s3_prefix: str) -> None:
    """Recursively upload all files in local_dir to s3://bucket/s3_prefix/."""
    uploaded = 0
    skipped = 0
    for root, _, files in os.walk(local_dir):
        for fname in files:
            local_path = os.path.join(root, fname)
            rel_path = os.path.relpath(local_path, local_dir)
            s3_key = f"{s3_prefix}/{rel_path}".replace("\\", "/")

            # Skip re-uploading if the object already exists with the same size
            try:
                head = s3_client.head_object(Bucket=bucket, Key=s3_key)
                remote_size = head["ContentLength"]
                local_size = os.path.getsize(local_path)
                if remote_size == local_size:
                    skipped += 1
                    continue
            except botocore.exceptions.ClientError:
                pass  # object doesn't exist — upload it

            print(f"  Uploading {local_path} -> s3://{bucket}/{s3_key}")
            s3_client.upload_file(local_path, bucket, s3_key)
            uploaded += 1

    print(f"  Done. Uploaded {uploaded} files, skipped {skipped} (already up-to-date).")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess datasets locally and upload to S3."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=ALL_DATASETS + ["all"],
        default=["all"],
        help="Datasets to process and upload (default: all).",
    )
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="Skip local preprocessing and only upload existing files.",
    )
    args = parser.parse_args()

    datasets = ALL_DATASETS if "all" in args.datasets else args.datasets

    boto_sess = boto3.Session(profile_name=AWS_PROFILE)
    s3_client = boto_sess.client("s3")

    # Verify bucket is accessible
    try:
        s3_client.head_bucket(Bucket=S3_BUCKET)
    except botocore.exceptions.ClientError as e:
        print(f"ERROR: Cannot access s3://{S3_BUCKET} — {e}")
        sys.exit(1)

    for dataset in datasets:
        local_dir = DATASET_LOCAL_DIRS[dataset]
        print(f"\n{'=' * 60}")
        print(f"Dataset: {dataset}")
        print(f"Local dir: {local_dir}")
        s3_dest = f"{S3_PREFIX}/{dataset}"
        print(f"S3 dest: s3://{S3_BUCKET}/{s3_dest}")
        print(f"{'=' * 60}")

        if not args.skip_preprocess:
            preprocess_dataset(dataset, local_dir)

        if not os.path.isdir(local_dir):
            print(f"  [WARN] Local directory '{local_dir}' not found, skipping upload.")
            continue

        upload_directory(s3_client, local_dir, S3_BUCKET, s3_dest)

    print("\nAll datasets processed and uploaded.")


if __name__ == "__main__":
    main()
