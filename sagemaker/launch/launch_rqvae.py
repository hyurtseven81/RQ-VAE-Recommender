"""Launch RQ-VAE training for a given dataset on SageMaker.

Usage::

    python sagemaker/launch/launch_rqvae.py --dataset toys
    python sagemaker/launch/launch_rqvae.py --dataset steam
    python sagemaker/launch/launch_rqvae.py --dataset beauty --instance-type ml.g5.2xlarge
    python sagemaker/launch/launch_rqvae.py --dataset sports \\
        --gin-config configs/rqvae_amazon_sports_v7.gin --job-suffix v7
"""
import argparse
from datetime import datetime, timezone

import boto3
import sagemaker
from sagemaker.inputs import TrainingInput
from sagemaker.pytorch import PyTorch


DATASETS = ["beauty", "sports", "toys", "steam"]
S3_BASE = "s3://REDACTED-BUCKET/rqvae-level-aware"


def _gin_config(dataset: str) -> str:
    if dataset == "steam":
        return "configs/rqvae_steam.gin"
    return f"configs/rqvae_amazon_{dataset}.gin"


def get_estimator(
    dataset: str,
    gin_config: str,
    output_subpath: str,
    instance_type: str,
    use_spot: bool,
    sess: sagemaker.Session,
) -> PyTorch:
    kwargs = dict(
        entry_point="train_rqvae.py",
        source_dir=".",
        role="arn:aws:iam::000000000000:role/REDACTED-ROLE",
        instance_type=instance_type,
        instance_count=1,
        framework_version="2.5.1",
        py_version="py311",
        sagemaker_session=sess,
        output_path=f"{S3_BASE}/rqvae/{output_subpath}/",
        hyperparameters={"config_path": gin_config},
        tags=[
            {"Key": "project", "Value": "rqvae-level-aware"},
            {"Key": "owner", "Value": "huseyin"},
        ],
    )
    if use_spot:
        kwargs.update(
            use_spot_instances=True,
            max_run=36000,
            max_wait=72000,
            checkpoint_s3_uri=f"{S3_BASE}/checkpoints/rqvae/{output_subpath}/",
        )
    else:
        kwargs.update(use_spot_instances=False, max_run=36000)
    return PyTorch(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch RQ-VAE training on SageMaker.")
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument(
        "--gin-config",
        default=None,
        help="Override the gin config path (e.g. configs/rqvae_amazon_sports_v7.gin).",
    )
    parser.add_argument(
        "--job-suffix",
        default=None,
        help="Optional tag appended to the job name and output subpath (e.g. 'v7').",
    )
    parser.add_argument("--instance-type", default="ml.g5.4xlarge")
    parser.add_argument(
        "--spot", action="store_true",
        help="Use spot instances (default off, since AGENTS.md flags spot as unreliable).",
    )
    parser.add_argument(
        "--dataset-s3",
        default=f"{S3_BASE}/datasets/amazon/",
        help="S3 URI with preprocessed dataset cache (passed as 'dataset' channel). "
             "Pass empty string to force fresh preprocessing inside the container.",
    )
    args = parser.parse_args()

    boto_sess = boto3.Session(profile_name="REDACTED-PROFILE", region_name="us-east-1")
    sess = sagemaker.Session(boto_session=boto_sess)

    gin_config = args.gin_config or _gin_config(args.dataset)
    output_subpath = f"{args.dataset}-{args.job_suffix}" if args.job_suffix else args.dataset
    estimator = get_estimator(
        args.dataset, gin_config, output_subpath, args.instance_type, args.spot, sess,
    )
    inputs = {}
    if args.dataset_s3:
        inputs["dataset"] = TrainingInput(args.dataset_s3)
    job_stub = f"rqvae-{args.dataset}"
    if args.job_suffix:
        job_stub += f"-{args.job_suffix}"
    job_name = f"{job_stub}-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
    estimator.fit(inputs=inputs, job_name=job_name, wait=False, logs=False)
    print(f"Launched: {job_name}")
    print(f"Output:   {S3_BASE}/rqvae/{output_subpath}/")


if __name__ == "__main__":
    main()
