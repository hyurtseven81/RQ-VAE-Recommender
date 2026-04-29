"""Launch MTL decoder training on SageMaker.

This is Stage 2 of `docs/paper_plan.md`. Trains a decoder with
`train_decoder_mtl.py` (joint SID + SASRec InfoNCE, grad-clip 1.0,
lambda warm-up) against the dataset's upstream pre-trained RQ-VAE
checkpoint. The resulting MTL decoder is the base for every
level-aware alpha experiment.

Usage::

    python sagemaker/launch/launch_mtl.py
    python sagemaker/launch/launch_mtl.py --datasets beauty sports ml32m
    python sagemaker/launch/launch_mtl.py --datasets beauty \\
        --pretrained-rqvae s3://bucket/prefix/rqvae-beauty-custom/output/model.tar.gz
"""
import argparse
from datetime import datetime, timezone

import boto3
from _aws_env import aws_profile, aws_region, s3_base, sagemaker_role
from sagemaker.inputs import TrainingInput
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
    use_spot: bool,
) -> PyTorch:
    # When --pretrained-rqvae is used, the shim in
    # modules/utils.override_save_dir_for_sagemaker() rebinds the gin path
    # to /opt/ml/input/data/model/*.pt, so no entry-point change is needed.
    kwargs = dict(
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
        hyperparameters={"config_path": _gin_config(dataset)},
        tags=[
            {"Key": "project", "Value": "rqvae-level-aware"},
            {"Key": "owner", "Value": "huseyin"},
            {"Key": "variant", "Value": "mtl"},
            {"Key": "paper-stage", "Value": "2"},
        ],
    )
    if use_spot:
        kwargs.update(use_spot_instances=True, max_run=72000, max_wait=144000)
    else:
        kwargs.update(use_spot_instances=False, max_run=72000)
    return PyTorch(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch MTL decoder training (Stage 2)."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASETS,
        default=DEFAULT_DATASETS,
        help="Datasets to launch (default: beauty + sports; add ml32m after "
             "Stage 0 gate).",
    )
    parser.add_argument(
        "--instance-type",
        default="ml.g5.2xlarge",
        help="SageMaker instance type (default: ml.g5.2xlarge).",
    )
    parser.add_argument(
        "--no-spot",
        action="store_true",
        help="Disable spot instances (spot is on by default, reliable enough "
             "for decoder training on g5.2xlarge).",
    )
    parser.add_argument(
        "--pretrained-rqvae",
        default=None,
        help="S3 URI of an RQ-VAE checkpoint (single dataset only — pair with "
             "a single --datasets value). When set, overrides the "
             "pretrained_rqvae_path in the gin config.",
    )
    parser.add_argument(
        "--dataset-s3",
        default=None,
        help="S3 URI with a preprocessed dataset cache (dataset channel). "
             "Defaults to $RQVAE_S3_BASE/datasets/amazon/ (or .../ml-32m/ for "
             "ml32m). Pass empty string to force fresh preprocessing.",
    )
    args = parser.parse_args()

    if args.pretrained_rqvae and len(args.datasets) != 1:
        parser.error(
            "--pretrained-rqvae is a single-checkpoint override; pair it with "
            "exactly one --datasets value."
        )

    boto_sess = boto3.Session(profile_name=aws_profile(), region_name=aws_region())
    sess = sagemaker.Session(boto_session=boto_sess)

    # Timestamp suffix so a relaunch doesn't collide with a prior Failed
    # job's name (SageMaker requires unique training-job names).
    stamp = f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"

    for dataset in args.datasets:
        estimator = get_estimator(dataset, args.instance_type, sess, use_spot=not args.no_spot)
        inputs = {}
        if args.pretrained_rqvae:
            inputs["model"] = TrainingInput(args.pretrained_rqvae)
        ds_s3 = args.dataset_s3
        if ds_s3 is None:
            ds_s3 = (
                f"{s3_base()}/datasets/ml-32m/"
                if dataset == "ml32m"
                else f"{s3_base()}/datasets/amazon/"
            )
        if ds_s3:
            inputs["dataset"] = TrainingInput(ds_s3)
        job_name = f"decoder-mtl-{dataset}-{stamp}"
        estimator.fit(
            inputs=inputs if inputs else None,
            job_name=job_name,
            wait=False,
            logs=False,
        )
        print(f"Launched MTL decoder job for dataset='{dataset}' -> job={job_name}")

    print(f"\nAll jobs submitted. Outputs at: {s3_base()}/decoder-mtl/")


if __name__ == "__main__":
    main()
