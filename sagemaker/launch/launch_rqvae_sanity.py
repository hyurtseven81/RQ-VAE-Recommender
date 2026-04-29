"""Launch short RQ-VAE sanity-check jobs on SageMaker.

For each requested dataset, runs `iterations` steps (default 5000) of training
against the gin config, then runs the validator and dumps verdict.json.
Uses the S3 preprocessed dataset cache so startup time is short.

Usage::

    python sagemaker/launch/launch_rqvae_sanity.py                # all 4 datasets
    python sagemaker/launch/launch_rqvae_sanity.py --datasets beauty sports
    python sagemaker/launch/launch_rqvae_sanity.py --iterations 10000

Bisect experiments (see docs/bisect_runbook.md) use the override flags::

    python sagemaker/launch/launch_rqvae_sanity.py --datasets beauty \\
        --gin-config configs/rqvae_amazon_beauty_bisect_a5367ed.gin \\
        --job-suffix bisect-a5367ed
    python sagemaker/launch/launch_rqvae_sanity.py --datasets beauty \\
        --disable-compile --job-suffix bisect-nocompile
"""
import argparse
from datetime import datetime, timezone

import boto3
from _aws_env import aws_profile, aws_region, s3_base, sagemaker_role
from sagemaker.inputs import TrainingInput
from sagemaker.pytorch import PyTorch

import sagemaker

DATASETS = ["beauty", "sports", "toys", "steam"]


def _gin_config(dataset: str) -> str:
    if dataset == "steam":
        return "configs/rqvae_steam.gin"
    return f"configs/rqvae_amazon_{dataset}.gin"


def _dataset_channel(dataset: str) -> str | None:
    """S3 URI for the preprocessed dataset cache. None if no cache exists yet."""
    if dataset in ("beauty", "sports", "toys"):
        return f"{s3_base()}/datasets/amazon/"
    if dataset == "steam":
        return f"{s3_base()}/datasets/steam/"  # may not exist yet
    return None


def get_estimator(
    dataset: str,
    gin_config: str,
    iterations: int,
    instance_type: str,
    sess: sagemaker.Session,
    output_subpath: str,
    disable_compile: bool = False,
) -> PyTorch:
    kwargs = dict(
        entry_point="sagemaker/rqvae_sanity_entry.py",
        source_dir=".",
        role=sagemaker_role(),
        instance_type=instance_type,
        instance_count=1,
        framework_version="2.5.1",
        py_version="py311",
        sagemaker_session=sess,
        output_path=f"{s3_base()}/rqvae-sanity/{output_subpath}/",
        use_spot_instances=False,
        max_run=5400,  # 90 min; covers container startup + 5k iters + validator
        hyperparameters={
            "config_path": gin_config,
            "iterations": iterations,
        },
        tags=[
            {"Key": "project", "Value": "rqvae-level-aware"},
            {"Key": "owner", "Value": "huseyin"},
            {"Key": "job-type", "Value": "rqvae-sanity"},
        ],
    )
    if disable_compile:
        kwargs["environment"] = {"RQVAE_DISABLE_COMPILE": "1"}
    return PyTorch(**kwargs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    ap.add_argument("--iterations", type=int, default=5000)
    ap.add_argument("--instance-type", default="ml.g5.4xlarge")
    ap.add_argument(
        "--gin-config",
        default=None,
        help="Override the gin config (e.g. a bisect variant). Applied to every "
             "dataset in --datasets; pair with a single-dataset invocation for "
             "per-variant A/B runs.",
    )
    ap.add_argument(
        "--job-suffix",
        default=None,
        help="Appended to the SageMaker job name and the S3 output subpath so "
             "bisect variants don't overwrite each other.",
    )
    ap.add_argument(
        "--disable-compile",
        action="store_true",
        help="Set RQVAE_DISABLE_COMPILE=1 in the container env so RqVae.forward "
             "runs eagerly instead of through torch.compile. Bisect diagnostic.",
    )
    args = ap.parse_args()

    boto_sess = boto3.Session(profile_name=aws_profile(), region_name=aws_region())
    sess = sagemaker.Session(boto_session=boto_sess)

    for dataset in args.datasets:
        gin_config = args.gin_config or _gin_config(dataset)
        output_subpath = (
            f"{dataset}-{args.job_suffix}" if args.job_suffix else dataset
        )
        estimator = get_estimator(
            dataset,
            gin_config,
            args.iterations,
            args.instance_type,
            sess,
            output_subpath,
            disable_compile=args.disable_compile,
        )
        inputs = {}
        ds_s3 = _dataset_channel(dataset)
        if ds_s3:
            inputs["dataset"] = TrainingInput(ds_s3)
        job_stub = f"rqvae-sanity-{dataset}"
        if args.job_suffix:
            job_stub += f"-{args.job_suffix}"
        job_name = f"{job_stub}-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
        estimator.fit(inputs=inputs, job_name=job_name, wait=False, logs=False)
        print(f"Launched: {job_name}")

    print(f"\nResults will land at: {s3_base()}/rqvae-sanity/")


if __name__ == "__main__":
    main()
