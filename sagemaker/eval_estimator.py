"""SageMaker estimator factory for inference-only evaluation jobs."""
import os

from sagemaker.pytorch import PyTorch

import sagemaker

_S3_BASE = os.environ.get("RQVAE_S3_BASE", "").rstrip("/")
if not _S3_BASE:
    raise RuntimeError(
        "RQVAE_S3_BASE not set. Example: "
        "export RQVAE_S3_BASE=s3://<your-bucket>/rqvae-level-aware"
    )


def get_estimator(
    dataset: str,
    decoder_checkpoint_s3: str,
    rqvae_checkpoint_s3: str,
    instance_type: str = "ml.g5.xlarge",
) -> PyTorch:
    """Create an estimator for running full held-out evaluation.

    Args:
        dataset: Dataset name (e.g. "beauty", "steam").
        decoder_checkpoint_s3: S3 URI of the decoder checkpoint to evaluate.
        rqvae_checkpoint_s3: S3 URI of the RQ-VAE checkpoint to use for tokenisation.
        instance_type: SageMaker instance type.
    """
    return PyTorch(
        entry_point="evaluate/run_eval.py",
        source_dir=".",
        role=sagemaker.get_execution_role(),
        instance_type=instance_type,
        instance_count=1,
        framework_version="2.5.1",
        py_version="py311",
        output_path=f"{_S3_BASE}/eval-results/{dataset}/",
        use_spot_instances=True,
        max_run=14400,
        max_wait=28800,
        hyperparameters={
            "gin_config": f"configs/decoder_{dataset}.gin",
            "decoder_checkpoint": decoder_checkpoint_s3,
            "rqvae_checkpoint": rqvae_checkpoint_s3,
        },
        tags=[
            {"Key": "project", "Value": "rqvae-level-aware"},
            {"Key": "owner", "Value": "huseyin"},
            {"Key": "job-type", "Value": "eval"},
        ],
    )
