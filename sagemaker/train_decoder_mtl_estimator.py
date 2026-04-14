"""SageMaker estimator factory for MTL (multi-task learning) decoder training jobs."""
import sagemaker
from sagemaker.pytorch import PyTorch


def get_estimator(dataset: str, instance_type: str = "ml.g5.2xlarge") -> PyTorch:
    return PyTorch(
        entry_point="train_decoder_mtl.py",
        source_dir=".",
        role=sagemaker.get_execution_role(),
        instance_type=instance_type,
        instance_count=1,
        framework_version="2.5.1",
        py_version="py311",
        output_path=f"s3://REDACTED-BUCKET/rqvae-level-aware/decoder-mtl/{dataset}/",
        checkpoint_s3_uri=f"s3://REDACTED-BUCKET/rqvae-level-aware/checkpoints/decoder-mtl/{dataset}/",
        use_spot_instances=True,
        max_run=72000,
        max_wait=144000,
        hyperparameters={
            "config_path": f"configs/decoder_amazon_{dataset}_mtl.gin"
            if dataset != "steam"
            else "configs/decoder_steam_mtl.gin",
        },
        tags=[
            {"Key": "project", "Value": "rqvae-level-aware"},
            {"Key": "owner", "Value": "huseyin"},
            {"Key": "variant", "Value": "mtl"},
        ],
    )
