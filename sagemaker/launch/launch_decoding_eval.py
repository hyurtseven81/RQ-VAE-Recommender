"""Launch held-out evaluation jobs on SageMaker.

One SageMaker job per (dataset, strategy) combination. This is the
execution path for both Stage 1 (alpha-free strategies) and Stage 2's
per-strategy comparison in `docs/paper_plan.md`.

Usage::

    # Stage 1 sweep — five alpha-free strategies across three datasets.
    python sagemaker/launch/launch_decoding_eval.py \\
        --datasets beauty sports ml32m \\
        --strategies vanilla dbs gumbel_topk hybrid \\
        --decoder-variant baseline

    # Single-shot: a specific (dataset, strategy, decoder-ckpt, rqvae-ckpt).
    python sagemaker/launch/launch_decoding_eval.py \\
        --datasets beauty \\
        --strategies level_aware_mix \\
        --decoder-variant mtl \\
        --alpha "0.3,0.5,0.1"

    # Let the launcher auto-discover ckpt URIs. For each dataset it picks the
    # most recently modified model.tar.gz under:
    #   baseline  -> $RQVAE_S3_BASE/decoder/<dataset>/
    #   mtl       -> $RQVAE_S3_BASE/decoder-mtl/<dataset>/
    # and the RQ-VAE ckpt from:
    #   $RQVAE_S3_BASE/checkpoints/rqvae_<dataset>_upstream/*.pt
    # Pass --decoder-ckpt / --rqvae-ckpt to override (only valid with a single
    # --datasets value).
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

# Must match keys in modules/decoding/__init__.py DECODING_STRATEGIES.
STRATEGIES = [
    # Stage 1 — alpha-free
    "vanilla", "dbs", "gumbel_topk", "hybrid", "sasrec_rerank",
    # Stage 2 — level-aware alpha
    "level_aware_mix", "level_aware_mix_grid", "level_aware_mix_learned",
]

DECODER_VARIANTS = ["baseline", "mtl"]

# Strategies that require a SASRec aux head → only an MTL decoder has it.
_MTL_REQUIRED = {
    "sasrec_rerank",
    "level_aware_mix",
    "level_aware_mix_grid",
    "level_aware_mix_learned",
}


def _gin_config(dataset: str, variant: str) -> str:
    suffix = "_mtl" if variant == "mtl" else ""
    if dataset == "steam":
        return f"configs/decoder_steam{suffix}.gin"
    if dataset == "ml32m":
        return f"configs/decoder_ml32m{suffix}.gin"
    if dataset == "ml1m":
        # Only an MTL config exists for ML1M currently.
        return "configs/decoder_ml1m_mtl.gin"
    # Amazon datasets: per-split MTL configs exist; for baseline fall back to
    # the generic Amazon template.
    if variant == "mtl":
        return f"configs/decoder_amazon_{dataset}_mtl.gin"
    return "configs/decoder_amazon.gin"


def _latest_artifact(s3_client, bucket: str, prefix: str, suffix: str) -> str | None:
    """Return the S3 URI of the most recently modified object under `prefix`
    whose key ends with `suffix`. Returns None if no match."""
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
    """Make a string safe for use in a SageMaker job name."""
    return re.sub(r"[^A-Za-z0-9-]+", "-", s).strip("-")


def _discover_decoder_ckpt(s3_client, bucket: str, dataset: str, variant: str) -> str | None:
    folder = "decoder" if variant == "baseline" else "decoder-mtl"
    prefix = f"rqvae-level-aware/{folder}/{dataset}/"
    return _latest_artifact(s3_client, bucket, prefix, "model.tar.gz")


def _discover_rqvae_ckpt(s3_client, bucket: str, dataset: str) -> str | None:
    """Prefer upstream-pinned checkpoint; fall back to fork trainings."""
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
    strategy: str,
    variant: str,
    decoder_ckpt: str,
    rqvae_ckpt: str,
    instance_type: str,
    sess: sagemaker.Session,
    alpha: str | None,
    alpha_ckpt_s3: str | None,
    alpha_csv_s3: str | None,
    use_spot: bool,
) -> PyTorch:
    # Channels mount under /opt/ml/input/data/<channel>/. run_eval._resolve_ckpt
    # handles the directory → .pt resolution and tar.gz extraction.
    hyperparameters = {
        "gin-config": _gin_config(dataset, variant),
        "decoder-checkpoint": "/opt/ml/input/data/decoder",
        "rqvae-checkpoint": "/opt/ml/input/data/rqvae",
        "dataset": dataset,
        "strategy": strategy,
        "output": "/opt/ml/output/data",
    }
    if alpha:
        hyperparameters["alpha"] = alpha
    if alpha_ckpt_s3:
        # run_eval.py reads alpha-ckpt as a local path; SageMaker will mount
        # the file via the `alpha_ckpt` channel. The entry sees
        # /opt/ml/input/data/alpha_ckpt/<filename>.
        hyperparameters["alpha-ckpt"] = "/opt/ml/input/data/alpha_ckpt"
    if alpha_csv_s3:
        hyperparameters["alpha-csv"] = "/opt/ml/input/data/alpha_csv"

    kwargs = dict(
        entry_point="evaluate/run_eval.py",
        source_dir=".",
        role=sagemaker_role(),
        instance_type=instance_type,
        instance_count=1,
        framework_version="2.5.1",
        py_version="py311",
        sagemaker_session=sess,
        output_path=f"{s3_base()}/eval-results/{variant}/{dataset}/{strategy}/",
        max_run=14400,
        hyperparameters=hyperparameters,
        tags=[
            {"Key": "project", "Value": "rqvae-level-aware"},
            {"Key": "owner", "Value": "huseyin"},
            {"Key": "job-type", "Value": "eval"},
            {"Key": "variant", "Value": variant},
            {"Key": "strategy", "Value": strategy},
        ],
    )
    if use_spot:
        kwargs.update(use_spot_instances=True, max_wait=28800)
    else:
        kwargs["use_spot_instances"] = False
    return PyTorch(**kwargs)


def main() -> None:
    p = argparse.ArgumentParser(description="Launch held-out evaluation jobs.")
    p.add_argument("--datasets", nargs="+", choices=DATASETS, default=["beauty", "sports", "ml32m"])
    p.add_argument("--strategies", nargs="+", choices=STRATEGIES,
                   default=["vanilla", "dbs", "gumbel_topk", "hybrid"],
                   help="Decoding strategies to evaluate (cartesian with --datasets).")
    p.add_argument("--decoder-variant", choices=DECODER_VARIANTS, default="baseline",
                   help="Which decoder to evaluate against. Stage 1 uses 'baseline'; "
                        "strategies that need a SASRec aux head force 'mtl'.")
    p.add_argument("--instance-type", default="ml.g5.xlarge")
    p.add_argument("--no-spot", action="store_true")
    p.add_argument("--decoder-ckpt", default=None,
                   help="Override decoder ckpt S3 URI (requires single --datasets).")
    p.add_argument("--rqvae-ckpt", default=None,
                   help="Override RQ-VAE ckpt S3 URI (requires single --datasets).")
    p.add_argument("--alpha", default=None,
                   help="Comma-separated per-level alpha for level_aware_mix "
                        "(e.g. '0.5,0.3,0.1'). Mutually exclusive with --alpha-ckpt.")
    p.add_argument("--alpha-ckpt", default=None,
                   help="S3 URI of AlphaParams .pt (for level_aware_mix_learned).")
    p.add_argument("--alpha-csv", default=None,
                   help="S3 URI of grid search CSV (for level_aware_mix_grid); "
                        "best alpha by recall@10 is used.")
    args = p.parse_args()

    if (args.decoder_ckpt or args.rqvae_ckpt) and len(args.datasets) != 1:
        p.error("--decoder-ckpt / --rqvae-ckpt require a single --datasets value.")

    # Auto-upgrade variant for strategies that need an MTL ckpt.
    effective_variant = args.decoder_variant
    if any(s in _MTL_REQUIRED for s in args.strategies) and effective_variant != "mtl":
        print(
            "[info] upgrading --decoder-variant to 'mtl' because at least one "
            "requested strategy needs a SASRec aux head."
        )
        effective_variant = "mtl"

    boto_sess = boto3.Session(profile_name=aws_profile(), region_name=aws_region())
    s3_client = boto_sess.client("s3")
    sess = sagemaker.Session(boto_session=boto_sess)
    bucket = s3_bucket()

    stamp = f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
    launched = 0

    for dataset in args.datasets:
        decoder_ckpt = args.decoder_ckpt or _discover_decoder_ckpt(
            s3_client, bucket, dataset, effective_variant
        )
        if decoder_ckpt is None:
            print(f"[SKIP] no {effective_variant} decoder ckpt for {dataset}")
            continue
        rqvae_ckpt = args.rqvae_ckpt or _discover_rqvae_ckpt(s3_client, bucket, dataset)
        if rqvae_ckpt is None:
            print(f"[SKIP] no RQ-VAE ckpt for {dataset}")
            continue

        for strategy in args.strategies:
            est = get_estimator(
                dataset, strategy, effective_variant, decoder_ckpt, rqvae_ckpt,
                args.instance_type, sess, args.alpha, args.alpha_ckpt,
                args.alpha_csv, use_spot=not args.no_spot,
            )
            inputs = {
                "decoder": TrainingInput(decoder_ckpt),
                "rqvae": TrainingInput(rqvae_ckpt),
            }
            if args.alpha_ckpt:
                inputs["alpha_ckpt"] = TrainingInput(args.alpha_ckpt)
            if args.alpha_csv:
                inputs["alpha_csv"] = TrainingInput(args.alpha_csv)

            job = _sanitize(f"eval-{effective_variant}-{dataset}-{strategy}-{stamp}")
            est.fit(inputs=inputs, job_name=job, wait=False, logs=False)
            print(f"Launched: {job}")
            print(f"  decoder={decoder_ckpt}")
            print(f"  rqvae  ={rqvae_ckpt}")
            launched += 1

    print(f"\nSubmitted {launched} eval job(s). Results at: "
          f"{s3_base()}/eval-results/{effective_variant}/")


if __name__ == "__main__":
    main()
