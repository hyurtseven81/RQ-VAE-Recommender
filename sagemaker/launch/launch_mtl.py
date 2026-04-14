"""Launch MTL decoder training for all four datasets on SageMaker.

Usage::

    python sagemaker/launch/launch_mtl.py
    python sagemaker/launch/launch_mtl.py --datasets beauty sports
    python sagemaker/launch/launch_mtl.py --instance-type ml.g5.4xlarge
"""
import argparse

import boto3
import sagemaker

from sagemaker.pytorch import PyTorch


DATASETS = ["beauty", "sports", "toys", "steam", "ml32m"]
S3_BASE = "s3://REDACTED-BUCKET/rqvae-level-aware"


def _gin_config(dataset: str) -> str:
    if dataset == "steam":
        return "configs/decoder_steam_mtl.gin"
    if dataset == "ml32m":
        return "configs/decoder_ml32m_mtl.gin"
    return f"configs/decoder_amazon_{dataset}_mtl.gin"


def get_estimator(
    dataset: str,
    instance_type: str,
    sess: sagemaker.Session,
) -> PyTorch:
    return PyTorch(
        entry_point="train_decoder_mtl.py",
        source_dir=".",
        role="arn:aws:iam::000000000000:role/REDACTED-ROLE",
        instance_type=instance_type,
        instance_count=1,
        framework_version="2.5.1",
        py_version="py311",
        sagemaker_session=sess,
        output_path=f"{S3_BASE}/decoder-mtl/{dataset}/",
        checkpoint_s3_uri=f"{S3_BASE}/checkpoints/decoder-mtl/{dataset}/",
        use_spot_instances=True,
        max_run=72000,
        max_wait=144000,
        hyperparameters={"config_path": _gin_config(dataset)},
        tags=[
            {"Key": "project", "Value": "rqvae-level-aware"},
            {"Key": "owner", "Value": "huseyin"},
            {"Key": "variant", "Value": "mtl"},
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch MTL decoder training for all datasets."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASETS,
        default=DATASETS,
        help="Datasets to launch (default: all four).",
    )
    parser.add_argument(
        "--instance-type",
        default="ml.g5.2xlarge",
        help="SageMaker instance type (default: ml.g5.2xlarge).",
    )
    args = parser.parse_args()

    boto_sess = boto3.Session(profile_name="REDACTED-PROFILE", region_name="us-east-1")
    sess = sagemaker.Session(boto_session=boto_sess)

    for dataset in args.datasets:
        estimator = get_estimator(dataset, args.instance_type, sess)
        estimator.fit(
            job_name=f"decoder-mtl-{dataset}",
            wait=False,
            logs=False,
        )
        print(f"Launched MTL decoder job for dataset='{dataset}'")

    print(f"\nAll jobs submitted. Monitor at: {S3_BASE}/decoder-mtl/")


if __name__ == "__main__":
    main()
