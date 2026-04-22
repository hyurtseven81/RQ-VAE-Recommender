"""Launch full held-out evaluation sweep across datasets and model variants.

For each (dataset, variant) pair the script checks whether a trained
decoder checkpoint exists in S3 before submitting the eval job.

Usage::

    python sagemaker/launch/launch_decoding_eval.py
    python sagemaker/launch/launch_decoding_eval.py --datasets beauty steam --variants baseline mtl
"""
import argparse

import boto3
from _aws_env import aws_profile, aws_region, s3_base, s3_bucket, sagemaker_role
from sagemaker.pytorch import PyTorch

import sagemaker

DATASETS = ["beauty", "sports", "toys", "steam"]
VARIANTS = ["baseline", "mtl"]
def _latest_checkpoint_uri(s3_client, bucket: str, prefix: str) -> str | None:
    """Return the S3 URI of the most recently modified .pt file under prefix."""
    paginator = s3_client.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
    candidates = []
    for page in pages:
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".pt"):
                candidates.append((obj["LastModified"], obj["Key"]))
    if not candidates:
        return None
    latest_key = max(candidates, key=lambda x: x[0])[1]
    return f"s3://{bucket}/{latest_key}"


def get_estimator(
    dataset: str,
    variant: str,
    decoder_checkpoint_s3: str,
    rqvae_checkpoint_s3: str,
    instance_type: str,
    sess: sagemaker.Session,
) -> PyTorch:
    variant_prefix = "" if variant == "baseline" else "_mtl"
    return PyTorch(
        entry_point="evaluate/run_eval.py",
        source_dir=".",
        role=sagemaker_role(),
        instance_type=instance_type,
        instance_count=1,
        framework_version="2.5.1",
        py_version="py311",
        sagemaker_session=sess,
        output_path=f"{s3_base()}/eval-results/{variant}/{dataset}/",
        use_spot_instances=True,
        max_run=14400,
        max_wait=28800,
        hyperparameters={
            "gin_config": f"configs/decoder{variant_prefix}_{dataset}.gin",
            "decoder_checkpoint": decoder_checkpoint_s3,
            "rqvae_checkpoint": rqvae_checkpoint_s3,
        },
        tags=[
            {"Key": "project", "Value": "rqvae-level-aware"},
            {"Key": "owner", "Value": "huseyin"},
            {"Key": "job-type", "Value": "eval"},
            {"Key": "variant", "Value": variant},
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch decoding evaluation sweep.")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=VARIANTS)
    parser.add_argument("--instance-type", default="ml.g5.xlarge")
    args = parser.parse_args()

    boto_sess = boto3.Session(profile_name=aws_profile(), region_name=aws_region())
    s3_client = boto_sess.client("s3")
    sess = sagemaker.Session(boto_session=boto_sess)
    bucket = s3_bucket()

    for dataset in args.datasets:
        rqvae_prefix = f"rqvae-level-aware/checkpoints/rqvae/{dataset}/"
        rqvae_ckpt = _latest_checkpoint_uri(s3_client, bucket, rqvae_prefix)
        if rqvae_ckpt is None:
            print(f"[SKIP] No RQ-VAE checkpoint found for dataset='{dataset}'. Train first.")
            continue

        for variant in args.variants:
            decoder_folder = "decoder" if variant == "baseline" else "decoder-mtl"
            decoder_prefix = f"rqvae-level-aware/checkpoints/{decoder_folder}/{dataset}/"
            decoder_ckpt = _latest_checkpoint_uri(s3_client, bucket, decoder_prefix)
            if decoder_ckpt is None:
                print(
                    f"[SKIP] No decoder checkpoint for dataset='{dataset}' variant='{variant}'."
                )
                continue

            estimator = get_estimator(
                dataset, variant, decoder_ckpt, rqvae_ckpt, args.instance_type, sess
            )
            estimator.fit(
                job_name=f"eval-{variant}-{dataset}",
                wait=False,
                logs=False,
            )
            print(f"Launched eval job: dataset='{dataset}', variant='{variant}'")

    print(f"\nAll eval jobs submitted. Results at: {s3_base()}/eval-results/")


if __name__ == "__main__":
    main()
