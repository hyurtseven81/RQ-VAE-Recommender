"""Shared AWS identifier lookup for SageMaker launch scripts.

Launchers read their AWS identifiers from environment variables so this
repository does not hard-code account-specific values. Required vars are
checked lazily (at call time, not import time), so importing this module
never fails in tests, CI, or IDE import scanners.

Required::

    RQVAE_S3_BASE            e.g. s3://<your-bucket>/rqvae-level-aware
    RQVAE_SAGEMAKER_ROLE     e.g. arn:aws:iam::<account-id>:role/<role-name>

Optional::

    RQVAE_AWS_PROFILE        boto3 profile name (falls back to AWS_PROFILE)
    RQVAE_AWS_REGION         region name      (falls back to AWS_REGION, then us-east-1)
"""
import os


def s3_base() -> str:
    v = os.environ.get("RQVAE_S3_BASE")
    if not v:
        raise RuntimeError(
            "RQVAE_S3_BASE not set. Example: "
            "export RQVAE_S3_BASE=s3://<your-bucket>/rqvae-level-aware"
        )
    return v.rstrip("/")


def s3_bucket() -> str:
    """Return just the bucket name from RQVAE_S3_BASE."""
    base = s3_base()
    assert base.startswith("s3://"), f"RQVAE_S3_BASE must start with s3:// (got {base!r})"
    return base.removeprefix("s3://").split("/", 1)[0]


def sagemaker_role() -> str:
    v = os.environ.get("RQVAE_SAGEMAKER_ROLE")
    if not v:
        raise RuntimeError(
            "RQVAE_SAGEMAKER_ROLE not set. Example: "
            "export RQVAE_SAGEMAKER_ROLE=arn:aws:iam::<account-id>:role/<role-name>"
        )
    return v


def aws_profile() -> str | None:
    """Return the boto3 profile name, or None to use the default credential chain."""
    return os.environ.get("RQVAE_AWS_PROFILE") or os.environ.get("AWS_PROFILE")


def aws_region() -> str:
    return os.environ.get("RQVAE_AWS_REGION") or os.environ.get("AWS_REGION", "us-east-1")
