"""SageMaker entry point: preprocess Amazon/Steam/ML32M datasets and copy output to SM_MODEL_DIR.

For --splits: amazon splits (beauty, sports, toys) and/or 'steam' and/or 'ml32m'.
Writes processed .pt files + raw data into SM_MODEL_DIR so SageMaker packs them
into model.tar.gz for extraction to s3://.../datasets/<dataset>/.

The companion sync step in `sagemaker/launch/launch_preprocess_datasets.py
--sync <job>` extracts the resulting tarball into the right
``$RQVAE_S3_BASE/datasets/<dataset>/`` prefix per dataset, which the
training launchers then mount as the ``dataset`` channel.
"""
import argparse
import os
import shutil
from pathlib import Path

VALID_SPLITS = ("beauty", "sports", "toys", "steam", "ml32m")


def _preprocess_amazon(local_root: Path, split: str) -> None:
    from data.amazon import AmazonReviews
    print(f"\n=== Preprocessing amazon split='{split}' ===", flush=True)
    AmazonReviews(root=str(local_root), split=split)
    print(f"Done: {split}", flush=True)


def _preprocess_steam(local_root: Path) -> None:
    from data.steam import RawSteam
    print("\n=== Preprocessing steam ===", flush=True)
    RawSteam(root=str(local_root))
    print("Done: steam", flush=True)


def _preprocess_ml32m(local_root: Path) -> None:
    from data.ml32m import RawMovieLens32M
    print("\n=== Preprocessing ml32m ===", flush=True)
    RawMovieLens32M(root=str(local_root))
    print("Done: ml32m", flush=True)


def _copy_to_sm(model_dir: Path, src_root: Path, dest_subdir: str) -> None:
    """Copy {processed, raw} subdirs of src_root into model_dir/<dest_subdir>/."""
    for sub in ("processed", "raw"):
        src = src_root / sub
        dest = model_dir / dest_subdir / sub
        if dest.exists():
            shutil.rmtree(dest)
        if src.exists():
            shutil.copytree(src, dest)
            print(f"Copied {src} -> {dest}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--splits", default="beauty,sports",
        help="Comma-separated subset of "
             "amazon splits (beauty,sports,toys), 'steam', and 'ml32m'.",
    )
    args = ap.parse_args()

    requested = [s.strip() for s in args.splits.split(",") if s.strip()]
    unknown = [s for s in requested if s not in VALID_SPLITS]
    if unknown:
        raise SystemExit(
            f"Unknown split(s): {unknown}. Valid: {list(VALID_SPLITS)}"
        )

    sm_model_dir = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    amazon_splits = [s for s in requested if s in ("beauty", "sports", "toys")]
    want_steam = "steam" in requested
    want_ml32m = "ml32m" in requested

    if amazon_splits:
        amazon_root = Path("dataset/amazon")
        amazon_root.mkdir(parents=True, exist_ok=True)
        for split in amazon_splits:
            _preprocess_amazon(amazon_root, split)
        _copy_to_sm(sm_model_dir, amazon_root, "amazon")

    if want_steam:
        steam_root = Path("dataset/steam")
        steam_root.mkdir(parents=True, exist_ok=True)
        _preprocess_steam(steam_root)
        _copy_to_sm(sm_model_dir, steam_root, "steam")

    if want_ml32m:
        ml32m_root = Path("dataset/ml-32m")
        ml32m_root.mkdir(parents=True, exist_ok=True)
        _preprocess_ml32m(ml32m_root)
        _copy_to_sm(sm_model_dir, ml32m_root, "ml-32m")


if __name__ == "__main__":
    main()
