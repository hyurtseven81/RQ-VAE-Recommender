"""Launch residual entropy analysis jobs for all trained RQ-VAE checkpoints.

Usage::

    python sagemaker/launch/launch_residual_entropy.py
    python sagemaker/launch/launch_residual_entropy.py --datasets beauty steam
"""
import argparse

import boto3
import sagemaker

from sagemaker.pytorch import PyTorch


DATASETS = ["beauty", "sports", "toys", "steam"]
S3_BASE = "s3://REDACTED-BUCKET/rqvae-level-aware"


def _latest_checkpoint_uri(s3_client, bucket: str, prefix: str) -> str | None:
    """Return the S3 URI of the most recently modified .pt file under prefix."""
    paginator = s3_client.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
    candidates = []
    for page in pages:
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".pt"):
                candidates.append((obj["LastModified"], obj["Key"]))
    if not candidates:
        return None
    latest_key = max(candidates, key=lambda x: x[0])[1]
    return f"s3://{bucket}/{latest_key}"


def get_estimator(
    dataset: str,
    rqvae_checkpoint_s3: str,
    instance_type: str,
    sess: sagemaker.Session,
) -> PyTorch:
    return PyTorch(
        entry_point="evaluate/residual_entropy.py",
        source_dir=".",
        role="arn:aws:iam::000000000000:role/REDACTED-ROLE",
        instance_type=instance_type,
        instance_count=1,
        framework_version="2.5.1",
        py_version="py311",
        sagemaker_session=sess,
        output_path=f"{S3_BASE}/residual-entropy/{dataset}/",
        use_spot_instances=True,
        max_run=7200,
        max_wait=14400,
        hyperparameters={
            "gin_config": f"configs/rqvae_{dataset}.gin",
            "rqvae_checkpoint": rqvae_checkpoint_s3,
        },
        tags=[
            {"Key": "project", "Value": "rqvae-level-aware"},
            {"Key": "owner", "Value": "huseyin"},
            {"Key": "job-type", "Value": "residual-entropy"},
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch residual entropy analysis.")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--instance-type", default="ml.g5.xlarge")
    args = parser.parse_args()

    boto_sess = boto3.Session(profile_name="REDACTED-PROFILE", region_name="us-east-1")
    s3_client = boto_sess.client("s3")
    sess = sagemaker.Session(boto_session=boto_sess)
    bucket = "REDACTED-BUCKET"

    for dataset in args.datasets:
        rqvae_prefix = f"rqvae-level-aware/checkpoints/rqvae/{dataset}/"
        rqvae_ckpt = _latest_checkpoint_uri(s3_client, bucket, rqvae_prefix)
        if rqvae_ckpt is None:
            print(f"[SKIP] No RQ-VAE checkpoint found for dataset='{dataset}'. Train first.")
            continue

        estimator = get_estimator(dataset, rqvae_ckpt, args.instance_type, sess)
        estimator.fit(
            job_name=f"residual-entropy-{dataset}",
            wait=False,
            logs=False,
        )
        print(f"Launched residual-entropy job for dataset='{dataset}'")

    print(f"\nAll jobs submitted. Results at: {S3_BASE}/residual-entropy/")


if __name__ == "__main__":
    main()
