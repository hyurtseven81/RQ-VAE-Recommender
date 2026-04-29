"""Deprecated — alpha grid search moved to ``evaluate/alpha_search.py``.

The earlier implementation in this file assumed a checkpoint format that
``train_decoder_mtl.py`` does not actually produce (expected full model
objects plus a ``codebook_embs`` key that is never saved) and emitted a
``recall_at_10`` column that would not sort against the real
``recall@K`` / ``ndcg@K`` schema from ``evaluate/metrics.py``. Use the
new entry point instead:

    # Local
    PYTHONPATH=. python evaluate/alpha_search.py \\
        --gin-config configs/decoder_amazon_beauty_mtl.gin \\
        --decoder-ckpt out/decoder_mtl/amazon_beauty/checkpoint_99999.pt \\
        --rqvae-ckpt   trained_models/rqvae_amazon_beauty/checkpoint_high_entropy.pt \\
        --dataset beauty \\
        --alpha-grid "0.0,0.5,1.0" \\
        --output results/stage2/beauty_pilot.csv

    # SageMaker (see sagemaker/launch/launch_alpha_search.py for the full CLI)
    python sagemaker/launch/launch_alpha_search.py --datasets beauty sports ml32m
"""
import sys


def main() -> None:
    sys.stderr.write(
        "scripts/alpha_grid_search.py has been removed. Use "
        "evaluate/alpha_search.py (or sagemaker/launch/launch_alpha_search.py) "
        "instead. See the module docstring for the new CLI.\n"
    )
    sys.exit(2)


if __name__ == "__main__":
    main()
