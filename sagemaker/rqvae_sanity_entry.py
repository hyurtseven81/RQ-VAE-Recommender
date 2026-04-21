"""SageMaker entry point: train RQ-VAE briefly, then run the validator.

Trains for `--iterations` (default 5000), writes a checkpoint to SM_MODEL_DIR,
loads it back through the validator, and dumps verdict.json alongside.
Downstream orchestration inspects verdict.json to decide whether to launch the
full 400k training run.
"""
import argparse
import json
import os
import sys

import gin

from evaluate.validate_rqvae import validate
from modules.utils import override_save_dir_for_sagemaker


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_path", required=True)
    ap.add_argument("--iterations", type=int, default=5000)
    args, _ = ap.parse_known_args()

    # Import first so @gin.configurable on train() registers before parse_config_file
    from train_rqvae import train

    gin.parse_config_file(args.config_path)
    override_save_dir_for_sagemaker()
    # Shrink the training run — keep a single end-of-run checkpoint
    gin.bind_parameter("train.iterations", args.iterations)
    gin.bind_parameter("train.save_model_every", args.iterations)
    gin.bind_parameter("train.eval_every", args.iterations)
    gin.bind_parameter("train.wandb_logging", False)

    print(f"=== Training for {args.iterations} iters with {args.config_path} ===", flush=True)
    train()
    print("=== Training done, running validator ===", flush=True)

    sm_model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    verdict = validate(
        config_path=args.config_path,
        rqvae_checkpoint=sm_model_dir,
        output_dir=sm_model_dir,
        parse_gin=False,  # gin already parsed above + mutated
    )
    print(json.dumps(verdict, indent=2))
    # Exit 0 regardless — verdict.json is the source of truth
    sys.exit(0)


if __name__ == "__main__":
    main()
