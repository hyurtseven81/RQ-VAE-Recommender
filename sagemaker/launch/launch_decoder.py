"""Launch vanilla SID-only decoder training on SageMaker.

This is Stage 1 of `docs/paper_plan.md`. Trains a decoder with
`train_decoder.py` (standard SID cross-entropy, no SASRec auxiliary head)
against the dataset's upstream pre-trained RQ-VAE checkpoint. The decoder
it produces is the reference point for all alpha-free beam-search
strategies.

Usage::

    python sagemaker/launch/launch_decoder.py                       # beauty + sports
    python sagemaker/launch/launch_decoder.py --datasets beauty
    python sagemaker/launch/launch_decoder.py --datasets beauty sports ml32m
    python sagemaker/launch/launch_decoder.py --datasets beauty \\
        --pretrained-rqvae s3://bucket/prefix/rqvae-beauty-custom/output/model.tar.gz

Note: the launcher uses upstream pre-trained RQ-VAE checkpoints by default
(paths in each `configs/decoder_*.gin`). Pass `--pretrained-rqvae` to
override with a specific S3 URI — SageMaker mounts it under the `model`
channel at `/opt/ml/input/data/model` and the launcher rewrites the gin
binding so `train_decoder.train.pretrained_rqvae_path` points there.
"""
import argparse

import boto3
from _aws_env import aws_profile, aws_region, s3_base, sagemaker_role
from sagemaker.inputs import TrainingInput
from sagemaker.pytorch import PyTorch

import sagemaker

DATASETS = ["beauty", "sports", "toys", "steam", "ml32m"]
DEFAULT_DATASETS = ["beauty", "sports"]


def _gin_config(dataset: str) -> str:
    if dataset == "steam":
        return "configs/decoder_steam.gin"
    if dataset == "ml32m":
        return "configs/decoder_ml32m.gin"
    return "configs/decoder_amazon.gin"


def get_estimator(
    dataset: str,
    instance_type: str,
    sess: sagemaker.Session,
    use_spot: bool,
) -> PyTorch:
    # SageMaker auto-mounts the `model` TrainingInput under
    # /opt/ml/input/data/model when --pretrained-rqvae is passed. The
    # container-side shim in `modules/utils.override_save_dir_for_sagemaker`
    # rebinds the gin `pretrained_rqvae_path` to that directory's .pt file
    # so no entry-point change is needed here.
    kwargs = dict(
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
        hyperparameters={"config_path": _gin_config(dataset)},
        tags=[
            {"Key": "project", "Value": "rqvae-level-aware"},
            {"Key": "owner", "Value": "huseyin"},
            {"Key": "variant", "Value": "vanilla"},
            {"Key": "paper-stage", "Value": "1"},
        ],
    )
    if use_spot:
        kwargs.update(
            use_spot_instances=True,
            max_run=72000,
            max_wait=144000,
        )
    else:
        kwargs.update(use_spot_instances=False, max_run=72000)
    return PyTorch(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch vanilla SID-only decoder training (Stage 1)."
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
        help="Disable spot instances. Default is spot on (g5.2xlarge spot is "
             "reliable enough for decoder training).",
    )
    parser.add_argument(
        "--pretrained-rqvae",
        default=None,
        help="S3 URI of an RQ-VAE checkpoint (single dataset only — pass with "
             "a single --datasets value). When set, overrides the "
             "pretrained_rqvae_path in the gin config by mounting the file "
             "via the 'model' channel.",
    )
    parser.add_argument(
        "--dataset-s3",
        default=None,
        help="S3 URI with a preprocessed dataset cache (passed as the "
             "'dataset' channel). Defaults to $RQVAE_S3_BASE/datasets/amazon/ "
             "for Amazon datasets and $RQVAE_S3_BASE/datasets/ml-32m/ for "
             "ml32m. Pass empty string to force fresh preprocessing inside "
             "the container.",
    )
    args = parser.parse_args()

    if args.pretrained_rqvae and len(args.datasets) != 1:
        parser.error(
            "--pretrained-rqvae is a single-checkpoint override; pair it with "
            "exactly one --datasets value."
        )

    boto_sess = boto3.Session(profile_name=aws_profile(), region_name=aws_region())
    sess = sagemaker.Session(boto_session=boto_sess)

    for dataset in args.datasets:
        estimator = get_estimator(
            dataset,
            args.instance_type,
            sess,
            use_spot=not args.no_spot,
        )
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
        job_name = f"decoder-{dataset}"
        estimator.fit(
            inputs=inputs if inputs else None,
            job_name=job_name,
            wait=False,
            logs=False,
        )
        print(f"Launched vanilla decoder job for dataset='{dataset}' -> job={job_name}")

    print(f"\nAll jobs submitted. Outputs at: {s3_base()}/decoder/")


if __name__ == "__main__":
    main()
