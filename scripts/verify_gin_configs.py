"""Quick verifier: does each requested gin config parse cleanly?

The Stage-1 / Stage-2 launchers all upload `source_dir="."` to SageMaker
which then `pip install -r requirements.txt` and runs the entry point in
the pinned container env. The training and eval scripts import wandb +
sentence_transformers at module scope, both of which can spuriously
block on a slow laptop network (HF Hub probe; wandb credential probe)
even though everything works fine inside the SageMaker container.

This script is the cheapest possible local pre-flight: it imports the
two `@gin.configurable` registry modules (`train_decoder`,
`train_decoder_mtl`) and tries to `parse_config_file` each requested
config. It does NOT import `evaluate.alpha_search` / `evaluate.run_eval`
or anything that pulls `sentence_transformers` / `data.processed`.

The script wall-clock is ~5 s on a cold venv. If the timeout below
trips, the laptop's network is the culprit, not the codebase — skip
the local check and let SageMaker be the real verifier.

Usage::

    python scripts/verify_gin_configs.py                # checks the 3 default Amazon decoder configs
    python scripts/verify_gin_configs.py configs/decoder_*_mtl.gin
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Disable network-touching subsystems before any heavy import. wandb
# probes credentials at import time on some versions; HF Hub does
# tokenizer-availability checks. Setting these here is cheap insurance
# against a hang regardless of what train_decoder / train_decoder_mtl
# transitively imports today or in the future.
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


_DEFAULT_CONFIGS = [
    "configs/decoder_amazon_beauty.gin",
    "configs/decoder_amazon_sports.gin",
    "configs/decoder_amazon_beauty_mtl.gin",
    "configs/decoder_amazon_sports_mtl.gin",
]


def _verify(configs: list[str]) -> int:
    """Return number of failures."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    import gin

    # Register both @gin.configurable decorators. Either may need to
    # resolve `train.*` (vanilla) or `train_mtl.*` (MTL) bindings.
    import train_decoder  # noqa: F401
    import train_decoder_mtl  # noqa: F401

    failures = 0
    for cfg in configs:
        gin.clear_config()
        try:
            gin.parse_config_file(cfg)
            print(f"  OK    {cfg}")
        except Exception as e:
            failures += 1
            print(f"  FAIL  {cfg}\n        {type(e).__name__}: {e}")
    return failures


def main() -> None:
    configs = sys.argv[1:] or _DEFAULT_CONFIGS
    missing = [c for c in configs if not Path(c).is_file()]
    if missing:
        print("ERROR: missing config file(s):", *missing, sep="\n  ")
        sys.exit(2)

    print(f"Verifying {len(configs)} gin config(s) "
          f"(WANDB_MODE={os.environ['WANDB_MODE']}, "
          f"HF_HUB_OFFLINE={os.environ['HF_HUB_OFFLINE']}):")
    n_fail = _verify(configs)

    if n_fail:
        print(f"\n{n_fail} config(s) failed to parse — see traces above.")
        sys.exit(1)
    print(f"\nAll {len(configs)} config(s) parse cleanly.")


if __name__ == "__main__":
    main()
