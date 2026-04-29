"""Launch vanilla decoder training for all four datasets on SageMaker.

This submits one job per dataset (beauty, sports, toys, steam) using the
baseline decoder configuration (no MTL, no level-aware decoding).

Usage::

    python sagemaker/launch/launch_baselines.py
    python sagemaker/launch/launch_baselines.py --datasets beauty sports
    python sagemaker/launch/launch_baselines.py --instance-type ml.g5.4xlarge
"""
import argparse

import boto3
from _aws_env import aws_profile, aws_region, s3_base, sagemaker_role
from sagemaker.pytorch import PyTorch

import sagemaker

DATASETS = ["beauty", "sports", "toys", "steam"]
def get_estimator(
    dataset: str,
    instance_type: str,
    sess: sagemaker.Session,
) -> PyTorch:
    return PyTorch(
        entry_point="train_decoder.py",
        source_dir=".",
        role=sagemaker_role(),
        instance_type=instance_type,
        instance_count=1,
        framework_version="2.5.1",
        py_version="py311",
        sagemaker_session=sess,
        output_path=f"{s3_base()}/decoder/{dataset}/",
        checkpoint_s3_uri=f"{s3_base()}/checkpoints/decoder/{dataset}/",
        use_spot_instances=True,
        max_run=72000,
        max_wait=144000,
        hyperparameters={"config_path": f"configs/decoder_{dataset}.gin"},
        tags=[
            {"Key": "project", "Value": "rqvae-level-aware"},
            {"Key": "owner", "Value": "huseyin"},
            {"Key": "variant", "Value": "baseline"},
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch baseline decoder training for all datasets."
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

    boto_sess = boto3.Session(profile_name=aws_profile(), region_name=aws_region())
    sess = sagemaker.Session(boto_session=boto_sess)

    for dataset in args.datasets:
        estimator = get_estimator(dataset, args.instance_type, sess)
        estimator.fit(
            job_name=f"decoder-baseline-{dataset}",
            wait=False,
            logs=False,
        )
        print(f"Launched baseline decoder job for dataset='{dataset}'")

    print(f"\nAll jobs submitted. Monitor at: {s3_base()}/decoder/")


if __name__ == "__main__":
    main()
