"""SageMaker entry point: preprocess Amazon datasets and copy output to SM_MODEL_DIR.

Runs AmazonReviews for the given splits, which triggers:
  1. Google Drive download of raw data (~1.2GB) into dataset/amazon/raw/<split>/
  2. Sentence-T5 embedding of item text (~40 min on ml.g5.4xlarge)
  3. Write of data_<split>.pt to dataset/amazon/processed/

Mirrors the result into SM_MODEL_DIR so SageMaker packs it to model.tar.gz,
which we then extract into s3://.../datasets/amazon/{processed,raw}/.
"""
import argparse
import os
import shutil
from pathlib import Path

from data.amazon import AmazonReviews


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="beauty,sports")
    args = ap.parse_args()

    sm_model_dir = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    local_root = Path("dataset/amazon")
    local_root.mkdir(parents=True, exist_ok=True)

    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        print(f"\n=== Preprocessing split='{split}' ===", flush=True)
        AmazonReviews(root=str(local_root), split=split)
        print(f"Done: {split}", flush=True)

    # Mirror processed + raw into SM_MODEL_DIR so SageMaker uploads as model.tar.gz
    for sub in ("processed", "raw"):
        src = local_root / sub
        dest = sm_model_dir / "amazon" / sub
        if dest.exists():
            shutil.rmtree(dest)
        if src.exists():
            shutil.copytree(src, dest)
            print(f"Copied {src} -> {dest}")


if __name__ == "__main__":
    main()
