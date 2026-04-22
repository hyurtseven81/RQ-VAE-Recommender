"""Launch RQ-VAE validation jobs on SageMaker (unique SIDs, residual entropy, min_dist).

Usage::

    python sagemaker/launch/launch_validate_rqvae.py                    # both Beauty + Sports v6
    python sagemaker/launch/launch_validate_rqvae.py --targets sports_v6
"""
import argparse

import boto3
import sagemaker
from sagemaker.inputs import TrainingInput
from sagemaker.pytorch import PyTorch


S3_BASE = "s3://REDACTED-BUCKET/rqvae-level-aware"

# Target spec: (job_name_stub, gin_config, checkpoint_s3_uri)
TARGETS = {
    "beauty": (
        "validate-rqvae-beauty",
        "configs/rqvae_amazon_beauty.gin",
        f"{S3_BASE}/checkpoints/rqvae_amazon_beauty/checkpoint_399999.pt",
    ),
    "sports_v6": (
        "validate-rqvae-sports-v6",
        "configs/rqvae_amazon_sports_v6.gin",
        f"{S3_BASE}/rqvae/sports-v6/rqvae-sports-v6-20260420-1856/output/model.tar.gz",
    ),
    "sports_v7": (
        "validate-rqvae-sports-v7",
        "configs/rqvae_amazon_sports_v7.gin",
        f"{S3_BASE}/rqvae/sports-v7/rqvae-sports-v7-20260421-101131/output/model.tar.gz",
    ),
    "beauty_repro": (
        "validate-rqvae-beauty-repro",
        "configs/rqvae_amazon_beauty.gin",
        f"{S3_BASE}/rqvae/beauty-repro/rqvae-beauty-repro-20260422-084727/output/model.tar.gz",
    ),
    "sports_repro": (
        "validate-rqvae-sports-repro",
        "configs/rqvae_amazon_sports.gin",
        f"{S3_BASE}/rqvae/sports-repro/rqvae-sports-repro-20260422-090910/output/model.tar.gz",
    ),
    "toys_repro": (
        "validate-rqvae-toys-repro",
        "configs/rqvae_amazon_toys.gin",
        f"{S3_BASE}/rqvae/toys-repro/rqvae-toys-repro-20260422-092032/output/model.tar.gz",
    ),
    "beauty_repro3": (
        "validate-rqvae-beauty-repro3",
        "configs/rqvae_amazon_beauty.gin",
        f"{S3_BASE}/rqvae/beauty-repro3/rqvae-beauty-repro3-20260422-112851/output/model.tar.gz",
    ),
    "sports_repro3": (
        "validate-rqvae-sports-repro3",
        "configs/rqvae_amazon_sports.gin",
        f"{S3_BASE}/rqvae/sports-repro3/rqvae-sports-repro3-20260422-113325/output/model.tar.gz",
    ),
    "toys_repro3": (
        "validate-rqvae-toys-repro3",
        "configs/rqvae_amazon_toys.gin",
        f"{S3_BASE}/rqvae/toys-repro3/rqvae-toys-repro3-20260422-113756/output/model.tar.gz",
    ),
    "steam_repro3": (
        "validate-rqvae-steam-repro3",
        "configs/rqvae_steam.gin",
        f"{S3_BASE}/rqvae/steam-repro3/rqvae-steam-repro3-20260422-120832/output/model.tar.gz",
    ),
}


def get_estimator(job_stub: str, gin_config: str, sess: sagemaker.Session,
                  instance_type: str) -> PyTorch:
    return PyTorch(
        entry_point="evaluate/validate_rqvae.py",
        source_dir=".",
        role="arn:aws:iam::000000000000:role/REDACTED-ROLE",
        instance_type=instance_type,
        instance_count=1,
        framework_version="2.5.1",
        py_version="py311",
        sagemaker_session=sess,
        output_path=f"{S3_BASE}/validate-rqvae/{job_stub}/",
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
        default=f"{S3_BASE}/datasets/amazon/",
        help="S3 URI with preprocessed dataset cache (passed as 'dataset' channel). "
             "Pass empty string to force fresh preprocessing inside the container.",
    )
    args = parser.parse_args()

    boto_sess = boto3.Session(profile_name="REDACTED-PROFILE", region_name="us-east-1")
    sess = sagemaker.Session(boto_session=boto_sess)

    for target in args.targets:
        job_stub, gin_config, ckpt_s3 = TARGETS[target]
        estimator = get_estimator(job_stub, gin_config, sess, args.instance_type)
        # SageMaker auto-extracts .tar.gz; a bare .pt is copied as-is.
        inputs = {"model": TrainingInput(ckpt_s3)}
        if args.dataset_s3:
            inputs["dataset"] = TrainingInput(args.dataset_s3)
        from datetime import datetime
        job_name = f"{job_stub}-{datetime.utcnow():%Y%m%d-%H%M%S}"
        estimator.fit(inputs=inputs, job_name=job_name, wait=False, logs=False)
        print(f"Launched: {job_name}  (ckpt={ckpt_s3})")

    print(f"\nResults will land at: {S3_BASE}/validate-rqvae/")


if __name__ == "__main__":
    main()
