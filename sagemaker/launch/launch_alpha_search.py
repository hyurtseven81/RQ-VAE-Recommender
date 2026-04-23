"""Launch alpha grid-search jobs on SageMaker (Stage 2 of the paper plan).

One SageMaker job per dataset. Each job builds the MTL decoder + RQ-VAE
once, then iterates a per-level alpha grid — much cheaper than spawning
one job per alpha point.

Usage::

    # Pilot — 3^3 = 27 alpha points per dataset.
    python sagemaker/launch/launch_alpha_search.py \\
        --datasets beauty sports ml32m \\
        --alpha-grid "0.0,0.5,1.0" \\
        --job-suffix pilot

    # Refined grid centred on each dataset's pilot winner (per-dataset grids).
    python sagemaker/launch/launch_alpha_search.py \\
        --datasets beauty \\
        --alpha0-grid "0.3,0.4,0.5,0.6,0.7" \\
        --alpha1-grid "0.3,0.4,0.5,0.6,0.7" \\
        --alpha2-grid "0.0,0.1,0.2,0.3,0.4" \\
        --job-suffix refined

The launcher auto-discovers ckpt URIs (MTL decoder + upstream RQ-VAE) from
S3. Pass `--decoder-ckpt` / `--rqvae-ckpt` to override with explicit URIs
(only valid with a single --datasets value).
"""
import argparse
import re
from datetime import datetime, timezone

import boto3
from _aws_env import aws_profile, aws_region, s3_base, s3_bucket, sagemaker_role
from sagemaker.inputs import TrainingInput
from sagemaker.pytorch import PyTorch

import sagemaker

DATASETS = ["beauty", "sports", "toys", "steam", "ml1m", "ml32m"]


def _gin_config(dataset: str) -> str:
    if dataset == "steam":
        return "configs/decoder_steam_mtl.gin"
    if dataset == "ml32m":
        return "configs/decoder_ml32m_mtl.gin"
    if dataset == "ml1m":
        return "configs/decoder_ml1m_mtl.gin"
    return f"configs/decoder_amazon_{dataset}_mtl.gin"


def _latest_artifact(s3_client, bucket: str, prefix: str, suffix: str) -> str | None:
    paginator = s3_client.get_paginator("list_objects_v2")
    candidates = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(suffix):
                candidates.append((obj["LastModified"], obj["Key"]))
    if not candidates:
        return None
    latest_key = max(candidates, key=lambda x: x[0])[1]
    return f"s3://{bucket}/{latest_key}"


def _sanitize(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9-]+", "-", s).strip("-")


def _discover_mtl_ckpt(s3_client, bucket: str, dataset: str) -> str | None:
    return _latest_artifact(
        s3_client, bucket, f"rqvae-level-aware/decoder-mtl/{dataset}/", "model.tar.gz"
    )


def _discover_rqvae_ckpt(s3_client, bucket: str, dataset: str) -> str | None:
    for prefix in (
        f"rqvae-level-aware/checkpoints/rqvae_{dataset}_upstream/",
        f"rqvae-level-aware/checkpoints/rqvae_amazon_{dataset}/",
        f"rqvae-level-aware/rqvae/{dataset}/",
    ):
        uri = _latest_artifact(s3_client, bucket, prefix, ".pt") or _latest_artifact(
            s3_client, bucket, prefix, "model.tar.gz"
        )
        if uri:
            return uri
    return None


def get_estimator(
    dataset: str,
    instance_type: str,
    sess: sagemaker.Session,
    alpha_grid: str | None,
    alpha_level_grids: dict[str, str],
    job_suffix: str,
    use_spot: bool,
) -> PyTorch:
    hyperparameters = {
        "gin-config": _gin_config(dataset),
        "decoder-checkpoint": "/opt/ml/input/data/decoder",
        "rqvae-checkpoint": "/opt/ml/input/data/rqvae",
        "dataset": dataset,
        "output": "/opt/ml/output/data",
        "job-name": _sanitize(f"alpha-{dataset}-{job_suffix}"),
    }
    if alpha_grid:
        hyperparameters["alpha-grid"] = alpha_grid
    for lvl_flag, spec in alpha_level_grids.items():
        if spec:
            hyperparameters[lvl_flag] = spec

    kwargs = dict(
        entry_point="evaluate/alpha_search.py",
        source_dir=".",
        role=sagemaker_role(),
        instance_type=instance_type,
        instance_count=1,
        framework_version="2.5.1",
        py_version="py311",
        sagemaker_session=sess,
        output_path=f"{s3_base()}/alpha-search/{dataset}-{job_suffix}/",
        max_run=28800,
        hyperparameters=hyperparameters,
        tags=[
            {"Key": "project", "Value": "rqvae-level-aware"},
            {"Key": "owner", "Value": "huseyin"},
            {"Key": "job-type", "Value": "alpha-search"},
            {"Key": "variant", "Value": job_suffix},
        ],
    )
    if use_spot:
        kwargs.update(use_spot_instances=True, max_wait=57600)
    else:
        kwargs["use_spot_instances"] = False
    return PyTorch(**kwargs)


def main() -> None:
    p = argparse.ArgumentParser(description="Launch per-level alpha grid search.")
    p.add_argument("--datasets", nargs="+", choices=DATASETS,
                   default=["beauty", "sports", "ml32m"])
    p.add_argument("--instance-type", default="ml.g5.xlarge")
    p.add_argument("--no-spot", action="store_true")
    p.add_argument("--job-suffix", default="pilot",
                   help="Appended to output path and job name so pilot/refined "
                        "runs don't collide (e.g. 'pilot', 'refined', 'cw01').")
    p.add_argument("--alpha-grid", default="0.0,0.5,1.0",
                   help="Shared grid applied to every level (cartesian). Ignored "
                        "for a level if --alphaN-grid is also passed.")
    p.add_argument("--alpha0-grid", default=None)
    p.add_argument("--alpha1-grid", default=None)
    p.add_argument("--alpha2-grid", default=None)
    p.add_argument("--decoder-ckpt", default=None,
                   help="Override MTL decoder S3 URI (requires single --datasets).")
    p.add_argument("--rqvae-ckpt", default=None,
                   help="Override RQ-VAE S3 URI (requires single --datasets).")
    args = p.parse_args()

    if (args.decoder_ckpt or args.rqvae_ckpt) and len(args.datasets) != 1:
        p.error("--decoder-ckpt / --rqvae-ckpt require a single --datasets value.")

    boto_sess = boto3.Session(profile_name=aws_profile(), region_name=aws_region())
    s3_client = boto_sess.client("s3")
    sess = sagemaker.Session(boto_session=boto_sess)
    bucket = s3_bucket()

    stamp = f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
    launched = 0
    for dataset in args.datasets:
        decoder_ckpt = args.decoder_ckpt or _discover_mtl_ckpt(s3_client, bucket, dataset)
        if decoder_ckpt is None:
            print(f"[SKIP] no MTL decoder ckpt for {dataset} under "
                  f"{s3_base()}/decoder-mtl/{dataset}/")
            continue
        rqvae_ckpt = args.rqvae_ckpt or _discover_rqvae_ckpt(s3_client, bucket, dataset)
        if rqvae_ckpt is None:
            print(f"[SKIP] no RQ-VAE ckpt for {dataset}")
            continue

        estimator = get_estimator(
            dataset,
            args.instance_type,
            sess,
            alpha_grid=args.alpha_grid,
            alpha_level_grids={
                "alpha0-grid": args.alpha0_grid,
                "alpha1-grid": args.alpha1_grid,
                "alpha2-grid": args.alpha2_grid,
            },
            job_suffix=args.job_suffix,
            use_spot=not args.no_spot,
        )
        inputs = {
            "decoder": TrainingInput(decoder_ckpt),
            "rqvae": TrainingInput(rqvae_ckpt),
        }
        job_name = _sanitize(f"alpha-{dataset}-{args.job_suffix}-{stamp}")
        estimator.fit(inputs=inputs, job_name=job_name, wait=False, logs=False)
        print(f"Launched: {job_name}")
        print(f"  decoder={decoder_ckpt}")
        print(f"  rqvae  ={rqvae_ckpt}")
        launched += 1

    print(f"\nSubmitted {launched} alpha-search job(s). "
          f"Results at: {s3_base()}/alpha-search/")


if __name__ == "__main__":
    main()
