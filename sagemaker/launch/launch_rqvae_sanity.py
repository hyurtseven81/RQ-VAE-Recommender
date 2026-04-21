"""Launch short RQ-VAE sanity-check jobs on SageMaker.

For each requested dataset, runs `iterations` steps (default 5000) of training
against the gin config, then runs the validator and dumps verdict.json.
Uses the S3 preprocessed dataset cache so startup time is short.

Usage::

    python sagemaker/launch/launch_rqvae_sanity.py                # all 4 datasets
    python sagemaker/launch/launch_rqvae_sanity.py --datasets beauty sports
    python sagemaker/launch/launch_rqvae_sanity.py --iterations 10000
"""
import argparse
from datetime import datetime, timezone

import boto3
import sagemaker
from sagemaker.inputs import TrainingInput
from sagemaker.pytorch import PyTorch


S3_BASE = "s3://REDACTED-BUCKET/rqvae-level-aware"
DATASETS = ["beauty", "sports", "toys", "steam"]


def _gin_config(dataset: str) -> str:
    if dataset == "steam":
        return "configs/rqvae_steam.gin"
    return f"configs/rqvae_amazon_{dataset}.gin"


def _dataset_channel(dataset: str) -> str | None:
    """S3 URI for the preprocessed dataset cache. None if no cache exists yet."""
    if dataset in ("beauty", "sports", "toys"):
        return f"{S3_BASE}/datasets/amazon/"
    if dataset == "steam":
        return f"{S3_BASE}/datasets/steam/"  # may not exist yet
    return None


def get_estimator(dataset: str, iterations: int, instance_type: str,
                  sess: sagemaker.Session) -> PyTorch:
    return PyTorch(
        entry_point="sagemaker/rqvae_sanity_entry.py",
        source_dir=".",
        role="arn:aws:iam::000000000000:role/REDACTED-ROLE",
        instance_type=instance_type,
        instance_count=1,
        framework_version="2.5.1",
        py_version="py311",
        sagemaker_session=sess,
        output_path=f"{S3_BASE}/rqvae-sanity/{dataset}/",
        use_spot_instances=False,
        max_run=5400,  # 90 min; covers container startup + 5k iters + validator
        hyperparameters={
            "config_path": _gin_config(dataset),
            "iterations": iterations,
        },
        tags=[
            {"Key": "project", "Value": "rqvae-level-aware"},
            {"Key": "owner", "Value": "huseyin"},
            {"Key": "job-type", "Value": "rqvae-sanity"},
        ],
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    ap.add_argument("--iterations", type=int, default=5000)
    ap.add_argument("--instance-type", default="ml.g5.4xlarge")
    args = ap.parse_args()

    boto_sess = boto3.Session(profile_name="REDACTED-PROFILE", region_name="us-east-1")
    sess = sagemaker.Session(boto_session=boto_sess)

    for dataset in args.datasets:
        estimator = get_estimator(dataset, args.iterations, args.instance_type, sess)
        inputs = {}
        ds_s3 = _dataset_channel(dataset)
        if ds_s3:
            inputs["dataset"] = TrainingInput(ds_s3)
        job_name = f"rqvae-sanity-{dataset}-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
        estimator.fit(inputs=inputs, job_name=job_name, wait=False, logs=False)
        print(f"Launched: {job_name}")

    print(f"\nResults will land at: {S3_BASE}/rqvae-sanity/")


if __name__ == "__main__":
    main()
