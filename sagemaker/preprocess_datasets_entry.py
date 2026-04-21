"""SageMaker entry point: preprocess Amazon/Steam datasets and copy output to SM_MODEL_DIR.

For --splits: amazon splits (beauty, sports, toys) and/or 'steam'.
Writes processed .pt files + raw data into SM_MODEL_DIR so SageMaker packs them
into model.tar.gz for extraction to s3://.../datasets/<dataset>/.
"""
import argparse
import os
import shutil
from pathlib import Path


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="beauty,sports",
                    help="Comma-separated: amazon splits (beauty,sports,toys) and/or 'steam'.")
    args = ap.parse_args()

    sm_model_dir = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))

    amazon_splits = [s.strip() for s in args.splits.split(",")
                     if s.strip() in ("beauty", "sports", "toys")]
    want_steam = "steam" in args.splits.split(",")

    if amazon_splits:
        amazon_root = Path("dataset/amazon")
        amazon_root.mkdir(parents=True, exist_ok=True)
        for split in amazon_splits:
            _preprocess_amazon(amazon_root, split)
        for sub in ("processed", "raw"):
            src = amazon_root / sub
            dest = sm_model_dir / "amazon" / sub
            if dest.exists():
                shutil.rmtree(dest)
            if src.exists():
                shutil.copytree(src, dest)
                print(f"Copied {src} -> {dest}")

    if want_steam:
        steam_root = Path("dataset/steam")
        steam_root.mkdir(parents=True, exist_ok=True)
        _preprocess_steam(steam_root)
        for sub in ("processed", "raw"):
            src = steam_root / sub
            dest = sm_model_dir / "steam" / sub
            if dest.exists():
                shutil.rmtree(dest)
            if src.exists():
                shutil.copytree(src, dest)
                print(f"Copied {src} -> {dest}")


if __name__ == "__main__":
    main()
