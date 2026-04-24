"""Launch RQ-VAE validation jobs on SageMaker (unique SIDs, residual entropy, min_dist).

Usage::

    python sagemaker/launch/launch_validate_rqvae.py                    # both Beauty + Sports v6
    python sagemaker/launch/launch_validate_rqvae.py --targets sports_v6
"""
import argparse

import boto3
from _aws_env import aws_profile, aws_region, s3_base, sagemaker_role
from sagemaker.inputs import TrainingInput
from sagemaker.pytorch import PyTorch

import sagemaker

# Target spec: (job_name_stub, gin_config, checkpoint_s3_suffix_under_RQVAE_S3_BASE)
# Suffixes are rendered against `s3_base()` at call time so this module imports
# even when RQVAE_S3_BASE is unset (e.g. --help, lint).
TARGETS = {
    "beauty": (
        "validate-rqvae-beauty",
        "configs/rqvae_amazon_beauty.gin",
        "checkpoints/rqvae_amazon_beauty/checkpoint_399999.pt",
    ),
    "sports_v6": (
        "validate-rqvae-sports-v6",
        "configs/rqvae_amazon_sports_v6.gin",
        "rqvae/sports-v6/rqvae-sports-v6-20260420-1856/output/model.tar.gz",
    ),
    "sports_v7": (
        "validate-rqvae-sports-v7",
        "configs/rqvae_amazon_sports_v7.gin",
        "rqvae/sports-v7/rqvae-sports-v7-20260421-101131/output/model.tar.gz",
    ),
    "beauty_repro": (
        "validate-rqvae-beauty-repro",
        "configs/rqvae_amazon_beauty.gin",
        "rqvae/beauty-repro/rqvae-beauty-repro-20260422-084727/output/model.tar.gz",
    ),
    "sports_repro": (
        "validate-rqvae-sports-repro",
        "configs/rqvae_amazon_sports.gin",
        "rqvae/sports-repro/rqvae-sports-repro-20260422-090910/output/model.tar.gz",
    ),
    "toys_repro": (
        "validate-rqvae-toys-repro",
        "configs/rqvae_amazon_toys.gin",
        "rqvae/toys-repro/rqvae-toys-repro-20260422-092032/output/model.tar.gz",
    ),
    "beauty_repro3": (
        "validate-rqvae-beauty-repro3",
        "configs/rqvae_amazon_beauty.gin",
        "rqvae/beauty-repro3/rqvae-beauty-repro3-20260422-112851/output/model.tar.gz",
    ),
    "sports_repro3": (
        "validate-rqvae-sports-repro3",
        "configs/rqvae_amazon_sports.gin",
        "rqvae/sports-repro3/rqvae-sports-repro3-20260422-113325/output/model.tar.gz",
    ),
    "toys_repro3": (
        "validate-rqvae-toys-repro3",
        "configs/rqvae_amazon_toys.gin",
        "rqvae/toys-repro3/rqvae-toys-repro3-20260422-113756/output/model.tar.gz",
    ),
    "steam_repro3": (
        "validate-rqvae-steam-repro3",
        "configs/rqvae_steam.gin",
        "rqvae/steam-repro3/rqvae-steam-repro3-20260422-120832/output/model.tar.gz",
    ),
    # --- Upstream pre-trained checkpoints (paper-plan Stage 0 gate) ---
    # These live at $RQVAE_S3_BASE/checkpoints/rqvae_<dataset>_upstream/ after
    # the `aws s3 cp` step in docs/runbook.md §0.2.
    "beauty_upstream": (
        "validate-rqvae-beauty-upstream",
        "configs/rqvae_amazon_beauty.gin",
        "checkpoints/rqvae_beauty_upstream/checkpoint_high_entropy.pt",
    ),
    "sports_upstream": (
        "validate-rqvae-sports-upstream",
        "configs/rqvae_amazon_sports.gin",
        "checkpoints/rqvae_sports_upstream/checkpoint_high_entropy.pt",
    ),
    "ml32m_upstream": (
        "validate-rqvae-ml32m-upstream",
        "configs/rqvae_ml32m.gin",
        "checkpoints/rqvae_ml32m_upstream/checkpoint_high_entropy.pt",
    ),
}


def get_estimator(job_stub: str, gin_config: str, sess: sagemaker.Session,
                  instance_type: str) -> PyTorch:
    return PyTorch(
        entry_point="evaluate/validate_rqvae.py",
        source_dir=".",
        role=sagemaker_role(),
        instance_type=instance_type,
        instance_count=1,
        framework_version="2.5.1",
        py_version="py311",
        sagemaker_session=sess,
        output_path=f"{s3_base()}/validate-rqvae/{job_stub}/",
        use_spot_instances=False,  # short job; on-demand avoids spot wait
        max_run=7200,  # 2h; data preprocessing runs fresh on each container
        hyperparameters={
            "config_path": gin_config,
            "rqvae_checkpoint": "/opt/ml/input/data/model",
        },
        tags=[
            {"Key": "project", "Value": "rqvae-level-aware"},
            {"Key": "owner", "Value": "huseyin"},
            {"Key": "job-type", "Value": "validate-rqvae"},
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--targets", nargs="+", choices=list(TARGETS), default=list(TARGETS)
    )
    parser.add_argument("--instance-type", default="ml.g5.4xlarge")
    parser.add_argument(
        "--dataset-s3",
        default=None,
        help="S3 URI with preprocessed dataset cache (passed as 'dataset' channel). "
             "Defaults to $RQVAE_S3_BASE/datasets/amazon/. Pass empty string to "
             "force fresh preprocessing inside the container.",
    )
    args = parser.parse_args()

    boto_sess = boto3.Session(profile_name=aws_profile(), region_name=aws_region())
    sess = sagemaker.Session(boto_session=boto_sess)

    dataset_s3 = args.dataset_s3 if args.dataset_s3 is not None else f"{s3_base()}/datasets/amazon/"
    for target in args.targets:
        job_stub, gin_config, ckpt_suffix = TARGETS[target]
        ckpt_s3 = f"{s3_base()}/{ckpt_suffix}"
        estimator = get_estimator(job_stub, gin_config, sess, args.instance_type)
        # SageMaker auto-extracts .tar.gz; a bare .pt is copied as-is.
        inputs = {"model": TrainingInput(ckpt_s3)}
        if dataset_s3:
            inputs["dataset"] = TrainingInput(dataset_s3)
        from datetime import datetime
        job_name = f"{job_stub}-{datetime.utcnow():%Y%m%d-%H%M%S}"
        estimator.fit(inputs=inputs, job_name=job_name, wait=False, logs=False)
        print(f"Launched: {job_name}  (ckpt={ckpt_s3})")

    print(f"\nResults will land at: {s3_base()}/validate-rqvae/")


if __name__ == "__main__":
    main()
