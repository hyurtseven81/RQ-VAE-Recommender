"""SageMaker estimator script for residual entropy analysis.

Loads a trained RQ-VAE checkpoint, runs the residual entropy analysis pipeline
over a dataset's item features, and writes the level-statistics parquet to S3
(via SageMaker's output channel).

Expected environment variables / SageMaker channel layout:
  SM_CHANNEL_MODEL  — directory containing the RQ-VAE checkpoint (checkpoint.pt)
  SM_CHANNEL_DATA   — directory containing item_features.pt  (FloatTensor [N, D])
  SM_OUTPUT_DATA_DIR — destination for output artifacts

Optional hyperparameters (passed via --<key> <value>):
  --batch-size   int   default 256
  --device       str   default "cpu"
  --alpha-schedule  comma-separated floats, e.g. "0.8,0.5,0.2"
"""
import argparse
import os
from pathlib import Path

import torch

from modules.analysis.residual_entropy import run_analysis
from modules.rqvae import RqVae


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--alpha-schedule",
        type=str,
        default=None,
        help="Comma-separated per-level alpha values, e.g. '0.8,0.5,0.2'",
    )
    # SageMaker injects these automatically
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_CHANNEL_MODEL", ""))
    parser.add_argument("--data-dir", type=str, default=os.environ.get("SM_CHANNEL_DATA", ""))
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # --- Load item features ---
    features_path = Path(args.data_dir) / "item_features.pt"
    if not features_path.exists():
        raise FileNotFoundError(f"Item features not found at {features_path}")
    item_features: torch.Tensor = torch.load(features_path, map_location="cpu")
    print(f"Loaded item features: {item_features.shape}")

    # --- Load RQ-VAE ---
    ckpt_path = Path(args.model_dir) / "checkpoint.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"RQ-VAE checkpoint not found at {ckpt_path}")
    state = torch.load(ckpt_path, map_location=args.device, weights_only=False)

    # RqVae.from_pretrained or reconstruct from saved config
    model_cfg = state.get("model_config", {})
    rqvae = RqVae(**model_cfg)
    rqvae.load_state_dict(state["model"])
    rqvae = rqvae.to(args.device)
    print("Loaded RQ-VAE checkpoint.")

    # --- Parse alpha schedule if provided ---
    alpha_schedule = None
    if args.alpha_schedule:
        alpha_schedule = [float(v) for v in args.alpha_schedule.split(",")]
        print(f"Using alpha schedule: {alpha_schedule}")

    # --- Run analysis ---
    output_path = str(Path(args.output_dir) / "residual_entropy_stats.parquet")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    run_analysis(
        rqvae=rqvae,
        item_features=item_features,
        output_path=output_path,
        alpha_schedule=alpha_schedule,
        batch_size=args.batch_size,
        device=args.device,
    )
    print(f"Analysis complete. Results written to {output_path}")


if __name__ == "__main__":
    main()
