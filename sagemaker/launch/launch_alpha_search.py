"""Launch Optuna alpha-search jobs for hybrid-decoding across datasets.

Usage::

    python sagemaker/launch/launch_alpha_search.py
    python sagemaker/launch/launch_alpha_search.py --datasets beauty steam --n-trials 100
"""
import argparse

import boto3
import sagemaker

from sagemaker.pytorch import PyTorch


DATASETS = ["beauty", "sports", "toys", "steam"]
S3_BASE = "s3://REDACTED-BUCKET/rqvae-level-aware"


def get_estimator(
    dataset: str,
    n_trials: int,
    instance_type: str,
    sess: sagemaker.Session,
) -> PyTorch:
    return PyTorch(
        entry_point="evaluate/alpha_search.py",
        source_dir=".",
        role="arn:aws:iam::000000000000:role/REDACTED-ROLE",
        instance_type=instance_type,
        instance_count=1,
        framework_version="2.5.1",
        py_version="py311",
        sagemaker_session=sess,
        output_path=f"{S3_BASE}/alpha-search/{dataset}/",
        use_spot_instances=True,
        max_run=14400,
        max_wait=28800,
        hyperparameters={
            "gin_config": f"configs/decoder_{dataset}.gin",
            "n_trials": n_trials,
        },
        tags=[
            {"Key": "project", "Value": "rqvae-level-aware"},
            {"Key": "owner", "Value": "huseyin"},
            {"Key": "job-type", "Value": "alpha-search"},
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch alpha hyper-parameter search.")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of Optuna trials per dataset (default: 50).",
    )
    parser.add_argument("--instance-type", default="ml.g5.xlarge")
    args = parser.parse_args()

    boto_sess = boto3.Session(profile_name="REDACTED-PROFILE", region_name="us-east-1")
    sess = sagemaker.Session(boto_session=boto_sess)

    for dataset in args.datasets:
        estimator = get_estimator(dataset, args.n_trials, args.instance_type, sess)
        estimator.fit(
            job_name=f"alpha-search-{dataset}",
            wait=False,
            logs=False,
        )
        print(f"Launched alpha-search job for dataset='{dataset}' (n_trials={args.n_trials})")

    print(f"\nAll jobs submitted. Results at: {S3_BASE}/alpha-search/")


if __name__ == "__main__":
    main()
