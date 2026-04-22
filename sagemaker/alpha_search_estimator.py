"""SageMaker estimator factory for Optuna alpha-search jobs."""
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
    n_trials: int = 50,
    instance_type: str = "ml.g5.xlarge",
) -> PyTorch:
    """Create an estimator for hybrid-decoding alpha hyper-parameter search.

    Args:
        dataset: Dataset name (e.g. "beauty", "steam").
        n_trials: Number of Optuna trials to run.
        instance_type: SageMaker instance type.
    """
    return PyTorch(
        entry_point="evaluate/alpha_search.py",
        source_dir=".",
        role=sagemaker.get_execution_role(),
        instance_type=instance_type,
        instance_count=1,
        framework_version="2.5.1",
        py_version="py311",
        output_path=f"{_S3_BASE}/alpha-search/{dataset}/",
        use_spot_instances=True,
        max_run=14400,
        max_wait=28800,
        hyperparameters={
            "gin_config": f"configs/decoder_{dataset}.gin",
            "n_trials": n_trials,
        },
        tags=[
            {"Key": "project", "Value": "rqvae-level-aware"},
            {"Key": "owner", "Value": "huseyin"},
            {"Key": "job-type", "Value": "alpha-search"},
        ],
    )
