"""Launch MTL decoder training for the CIKM 2026 datasets on SageMaker.

Default scope is ``beauty`` + ``sports`` only — the remaining datasets
(toys, steam, ml32m) were dropped from the paper due to codebook-collapse
or data-pipeline incompatibilities (see AGENTS.md "Checkpoint status").
They remain available via ``--datasets`` for ad-hoc experiments but will
fail at checkpoint load unless the corresponding RQ-VAE ckpt exists.

Usage::

    python sagemaker/launch/launch_mtl.py
    python sagemaker/launch/launch_mtl.py --datasets beauty
    python sagemaker/launch/launch_mtl.py --instance-type ml.g5.4xlarge
"""
import argparse

import boto3
from _aws_env import aws_profile, aws_region, s3_base, sagemaker_role
from sagemaker.pytorch import PyTorch

import sagemaker

DATASETS = ["beauty", "sports", "toys", "steam", "ml32m"]
DEFAULT_DATASETS = ["beauty", "sports"]
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
        role=sagemaker_role(),
        instance_type=instance_type,
        instance_count=1,
        framework_version="2.5.1",
        py_version="py311",
        sagemaker_session=sess,
        output_path=f"{s3_base()}/decoder-mtl/{dataset}/",
        checkpoint_s3_uri=f"{s3_base()}/checkpoints/decoder-mtl/{dataset}/",
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
        description="Launch MTL decoder training for CIKM 2026 datasets."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASETS,
        default=DEFAULT_DATASETS,
        help="Datasets to launch (default: beauty + sports).",
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
            job_name=f"decoder-mtl-{dataset}",
            wait=False,
            logs=False,
        )
        print(f"Launched MTL decoder job for dataset='{dataset}'")

    print(f"\nAll jobs submitted. Monitor at: {s3_base()}/decoder-mtl/")


if __name__ == "__main__":
    main()
