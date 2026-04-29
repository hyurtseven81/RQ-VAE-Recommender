"""Deprecated — learned-alpha training moved to ``evaluate/alpha_train.py``.

The earlier implementation in this file assumed a checkpoint format that
``train_decoder_mtl.py`` does not actually produce (expected full model
objects and a ``codebook_embs`` key that is never saved). The replacement
under ``evaluate/alpha_train.py`` reconstructs the model via the same
``_setup_training`` helper used by the rest of the pipeline and loads
the MTL decoder state-dict via the same path ``evaluate/run_eval.py``
uses, so training-time and inference-time mixing are byte-for-byte
consistent.

Use the new entry point directly, or the SageMaker launcher:

    # Local
    PYTHONPATH=. python evaluate/alpha_train.py \\
        --gin-config configs/decoder_amazon_beauty_mtl.gin \\
        --decoder-ckpt out/decoder_mtl/amazon_beauty/checkpoint_99999.pt \\
        --rqvae-ckpt   trained_models/rqvae_amazon_beauty/checkpoint_high_entropy.pt \\
        --dataset beauty \\
        --output out/alpha_params/beauty_learned.pt

    # SageMaker (see sagemaker/launch/launch_alpha_params.py for the full CLI)
    python sagemaker/launch/launch_alpha_params.py --datasets beauty sports ml32m
"""
import sys


def main() -> None:
    sys.stderr.write(
        "scripts/train_alpha_params.py has been removed. Use "
        "evaluate/alpha_train.py (or sagemaker/launch/launch_alpha_params.py) "
        "instead. See the module docstring for the new CLI.\n"
    )
    sys.exit(2)


if __name__ == "__main__":
    main()
