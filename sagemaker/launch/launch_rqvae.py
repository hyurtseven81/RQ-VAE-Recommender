"""Launch RQ-VAE training for a given dataset on SageMaker.

Usage::

    python sagemaker/launch/launch_rqvae.py --dataset toys
    python sagemaker/launch/launch_rqvae.py --dataset steam
    python sagemaker/launch/launch_rqvae.py --dataset beauty --instance-type ml.g5.2xlarge
"""
import argparse

import boto3
import sagemaker

from sagemaker.pytorch import PyTorch


DATASETS = ["beauty", "sports", "toys", "steam"]
S3_BASE = "s3://REDACTED-BUCKET/rqvae-level-aware"


def _gin_config(dataset: str) -> str:
    if dataset == "steam":
        return "configs/rqvae_steam.gin"
    return f"configs/rqvae_amazon_{dataset}.gin"


def get_estimator(
    dataset: str,
    instance_type: str,
    sess: sagemaker.Session,
) -> PyTorch:
    return PyTorch(
        entry_point="train_rqvae.py",
        source_dir=".",
        role="arn:aws:iam::000000000000:role/REDACTED-ROLE",
        instance_type=instance_type,
        instance_count=1,
        framework_version="2.5.1",
        py_version="py311",
        sagemaker_session=sess,
        output_path=f"{S3_BASE}/rqvae/{dataset}/",
        checkpoint_s3_uri=f"{S3_BASE}/checkpoints/rqvae/{dataset}/",
        use_spot_instances=True,
        max_run=36000,
        max_wait=72000,
        hyperparameters={"config_path": _gin_config(dataset)},
        tags=[
            {"Key": "project", "Value": "rqvae-level-aware"},
            {"Key": "owner", "Value": "huseyin"},
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch RQ-VAE training on SageMaker.")
    parser.add_argument(
        "--dataset",
        choices=DATASETS,
        required=True,
        help="Dataset to train on.",
    )
    parser.add_argument(
        "--instance-type",
        default="ml.g5.xlarge",
        help="SageMaker instance type (default: ml.g5.xlarge).",
    )
    args = parser.parse_args()

    boto_sess = boto3.Session(profile_name="REDACTED-PROFILE", region_name="us-east-1")
    sess = sagemaker.Session(boto_session=boto_sess)

    estimator = get_estimator(args.dataset, args.instance_type, sess)
    estimator.fit(
        job_name=f"rqvae-{args.dataset}",
        wait=False,
        logs=False,
    )
    print(f"Launched RQ-VAE training job for dataset='{args.dataset}'")
    print(f"Output path: {S3_BASE}/rqvae/{args.dataset}/")


if __name__ == "__main__":
    main()
