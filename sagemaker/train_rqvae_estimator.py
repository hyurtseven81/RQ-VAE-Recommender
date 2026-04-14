"""SageMaker estimator factory for RQ-VAE training jobs."""
import sagemaker
from sagemaker.pytorch import PyTorch


def get_estimator(dataset: str, instance_type: str = "ml.g5.xlarge") -> PyTorch:
    return PyTorch(
        entry_point="train_rqvae.py",
        source_dir=".",
        role=sagemaker.get_execution_role(),
        instance_type=instance_type,
        instance_count=1,
        framework_version="2.5.1",
        py_version="py311",
        output_path=f"s3://REDACTED-BUCKET/rqvae-level-aware/rqvae/{dataset}/",
        checkpoint_s3_uri=f"s3://REDACTED-BUCKET/rqvae-level-aware/checkpoints/rqvae/{dataset}/",
        use_spot_instances=True,
        max_run=36000,
        max_wait=72000,
        hyperparameters={
            "config_path": f"configs/rqvae_amazon_{dataset}.gin"
            if dataset != "steam"
            else "configs/rqvae_steam.gin",
        },
        tags=[
            {"Key": "project", "Value": "rqvae-level-aware"},
            {"Key": "owner", "Value": "huseyin"},
        ],
    )
