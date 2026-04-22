"""Launch a one-shot SageMaker job to preprocess Amazon datasets and upload to S3.

The job runs Sentence-T5 text embedding for the selected splits, writes the
processed .pt files plus raw data into SM_MODEL_DIR, and SageMaker packs them
into model.tar.gz. A second pass in this launcher (``--sync``) can then be
used to extract that tarball into $RQVAE_S3_BASE/datasets/amazon/
so downstream jobs can mount it as a dataset channel and skip preprocessing.

Usage::

    # Launch preprocessing (fire-and-forget):
    python sagemaker/launch/launch_preprocess_datasets.py

    # After the job completes, sync the output tarball into datasets/amazon/:
    python sagemaker/launch/launch_preprocess_datasets.py --sync <job_name>
"""
import argparse
import io
import sys
import tarfile
from datetime import datetime, timezone

import boto3
from _aws_env import aws_profile, aws_region, s3_base, s3_bucket, sagemaker_role
from sagemaker.pytorch import PyTorch

import sagemaker

S3_DATASETS_PREFIX_ROOT = "rqvae-level-aware/datasets"


def launch(splits: str, instance_type: str) -> str:
    boto_sess = boto3.Session(profile_name=aws_profile(), region_name=aws_region())
    sess = sagemaker.Session(boto_session=boto_sess)

    job_name = f"preprocess-amazon-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
    estimator = PyTorch(
        entry_point="sagemaker/preprocess_datasets_entry.py",
        source_dir=".",
        role=sagemaker_role(),
        instance_type=instance_type,
        instance_count=1,
        framework_version="2.5.1",
        py_version="py311",
        sagemaker_session=sess,
        output_path=f"{s3_base()}/preprocess-datasets/",
        use_spot_instances=False,
        max_run=7200,
        hyperparameters={"splits": splits},
        tags=[
            {"Key": "project", "Value": "rqvae-level-aware"},
            {"Key": "owner", "Value": "huseyin"},
            {"Key": "job-type", "Value": "preprocess-datasets"},
        ],
    )
    estimator.fit(job_name=job_name, wait=False, logs=False)
    print(f"Launched: {job_name}")
    print("Monitor via SageMaker console or:")
    print(f"  aws sagemaker describe-training-job --training-job-name {job_name} --profile $RQVAE_AWS_PROFILE")
    print("Once Completed, run:")
    print(f"  python sagemaker/launch/launch_preprocess_datasets.py --sync {job_name}")
    return job_name


def sync(job_name: str) -> None:
    """Stream the model.tar.gz for the completed job and upload each entry to
    $RQVAE_S3_BASE/datasets/amazon/{processed,raw}/..."""
    boto_sess = boto3.Session(profile_name=aws_profile(), region_name=aws_region())
    s3 = boto_sess.client("s3")
    sm = boto_sess.client("sagemaker")

    job = sm.describe_training_job(TrainingJobName=job_name)
    if job["TrainingJobStatus"] != "Completed":
        print(f"Job status is {job['TrainingJobStatus']}, not Completed. Aborting.")
        sys.exit(2)
    tar_uri = job["ModelArtifacts"]["S3ModelArtifacts"]
    print(f"Downloading {tar_uri}")
    # Parse s3://bucket/key
    assert tar_uri.startswith("s3://")
    bucket, key = tar_uri[5:].split("/", 1)
    obj = s3.get_object(Bucket=bucket, Key=key)
    stream = io.BytesIO(obj["Body"].read())
    print("Extracting + uploading entries ...")
    uploaded = 0
    with tarfile.open(fileobj=stream, mode="r:gz") as tf:
        for member in tf:
            if not member.isfile():
                continue
            # Tarball entries look like "amazon/processed/data_beauty.pt" or
            # "steam/processed/data_steam.pt". Route each to the corresponding
            # datasets/<dataset>/ prefix.
            rel = member.name.removeprefix("./")
            parts = rel.split("/", 1)
            if len(parts) < 2:
                continue
            ds, sub = parts
            if ds not in ("amazon", "steam"):
                continue
            s3_key = f"{S3_DATASETS_PREFIX_ROOT}/{ds}/{sub}"
            data = tf.extractfile(member)
            if data is None:
                continue
            s3.upload_fileobj(data, s3_bucket(), s3_key)
            uploaded += 1
            if uploaded % 10 == 0:
                print(f"  uploaded {uploaded} files...")
    print(f"Done. Uploaded {uploaded} files to s3://{s3_bucket()}/{S3_DATASETS_PREFIX_ROOT}/")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="beauty,sports")
    ap.add_argument("--instance-type", default="ml.g5.4xlarge")
    ap.add_argument("--sync", metavar="JOB_NAME",
                    help="Skip launch; download model.tar.gz from the given completed "
                         "job and upload its contents to datasets/amazon/.")
    args = ap.parse_args()

    if args.sync:
        sync(args.sync)
    else:
        launch(args.splits, args.instance_type)


if __name__ == "__main__":
    main()
