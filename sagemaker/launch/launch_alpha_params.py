"""Launch learned-alpha training jobs on SageMaker (Stage 2 — learned variant).

One SageMaker job per dataset. Each job loads the dataset's MTL decoder +
RQ-VAE, freezes them, and trains an ``AlphaParams`` module via the
teacher-forced mixed cross-entropy loss (see ``evaluate/alpha_train.py``).
The resulting ``*.pt`` can be fed into
``sagemaker/launch/launch_decoding_eval.py --alpha-ckpt ...`` with the
``level_aware_mix_learned`` strategy.

Usage::

    python sagemaker/launch/launch_alpha_params.py \\
        --datasets beauty sports ml32m \\
        --init-alpha "0.5,0.5,0.5" \\
        --n-epochs 1

    # Override ckpts for a one-off dataset:
    python sagemaker/launch/launch_alpha_params.py \\
        --datasets beauty \\
        --decoder-ckpt s3://bucket/prefix/decoder-mtl/beauty/<job>/output/model.tar.gz \\
        --rqvae-ckpt   s3://bucket/prefix/checkpoints/rqvae_beauty_upstream/checkpoint_high_entropy.pt

Defaults:
  - on-demand ``ml.g5.2xlarge`` (gradient-state matters; spot interruptions
    would cost progress).
  - 1 epoch, lr=0.05, init alpha=sigmoid(0)=0.5 per level.
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
    init_alpha: str | None,
    lr: float,
    n_epochs: int,
    max_steps: int,
    job_suffix: str,
    use_spot: bool,
) -> PyTorch:
    hyperparameters = {
        "gin-config": _gin_config(dataset),
        "decoder-checkpoint": "/opt/ml/input/data/decoder",
        "rqvae-checkpoint": "/opt/ml/input/data/rqvae",
        "dataset": dataset,
        "output": f"/opt/ml/model/{dataset}_learned.pt",
        "lr": str(lr),
        "n-epochs": str(n_epochs),
        "job-name": _sanitize(f"alpha-learned-{dataset}-{job_suffix}"),
    }
    if init_alpha:
        hyperparameters["init-alpha"] = init_alpha
    if max_steps > 0:
        hyperparameters["max-steps"] = str(max_steps)

    kwargs = dict(
        entry_point="evaluate/alpha_train.py",
        source_dir=".",
        role=sagemaker_role(),
        instance_type=instance_type,
        instance_count=1,
        framework_version="2.5.1",
        py_version="py311",
        sagemaker_session=sess,
        output_path=f"{s3_base()}/alpha-learned/{dataset}-{job_suffix}/",
        checkpoint_s3_uri=f"{s3_base()}/checkpoints/alpha-learned/{dataset}-{job_suffix}/",
        max_run=28800,
        hyperparameters=hyperparameters,
        tags=[
            {"Key": "project", "Value": "rqvae-level-aware"},
            {"Key": "owner", "Value": "huseyin"},
            {"Key": "job-type", "Value": "alpha-learned"},
            {"Key": "variant", "Value": job_suffix},
        ],
    )
    if use_spot:
        # On-demand default — gradient state is non-trivial and a spot
        # preemption loses the optimiser state entirely. Pass --spot
        # explicitly if you want to opt in.
        kwargs.update(use_spot_instances=True, max_wait=57600)
    else:
        kwargs["use_spot_instances"] = False
    return PyTorch(**kwargs)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Launch learned-alpha training (Stage 2, learned variant)."
    )
    p.add_argument("--datasets", nargs="+", choices=DATASETS,
                   default=["beauty", "sports", "ml32m"])
    p.add_argument("--instance-type", default="ml.g5.2xlarge")
    p.add_argument("--spot", action="store_true",
                   help="Opt in to spot instances (default off — gradient state "
                        "lost on preemption).")
    p.add_argument("--job-suffix", default="learned")
    p.add_argument("--init-alpha", default=None,
                   help="Comma-separated initial alpha per level (e.g. '0.5,0.5,0.5'). "
                        "Default sigmoid(0)=0.5 per level.")
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--n-epochs", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=0,
                   help="Cap training steps per epoch (useful for quick runs). "
                        "0 = full epoch.")
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

        est = get_estimator(
            dataset, args.instance_type, sess,
            init_alpha=args.init_alpha, lr=args.lr,
            n_epochs=args.n_epochs, max_steps=args.max_steps,
            job_suffix=args.job_suffix, use_spot=args.spot,
        )
        inputs = {
            "decoder": TrainingInput(decoder_ckpt),
            "rqvae": TrainingInput(rqvae_ckpt),
        }
        job_name = _sanitize(f"alpha-learned-{dataset}-{args.job_suffix}-{stamp}")
        est.fit(inputs=inputs, job_name=job_name, wait=False, logs=False)
        print(f"Launched: {job_name}")
        print(f"  decoder={decoder_ckpt}")
        print(f"  rqvae  ={rqvae_ckpt}")
        launched += 1

    print(f"\nSubmitted {launched} learned-alpha job(s). "
          f"Outputs at: {s3_base()}/alpha-learned/")


if __name__ == "__main__":
    main()
