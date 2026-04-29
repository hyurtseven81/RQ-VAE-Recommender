"""Diagnostic: compare train-time vs eval-time corpus_ids for a decoder ckpt.

Designed for the §M-zero-metrics decision-tree row 1 case ("training-time
eval works, offline eval is broken for the same ckpt"): the symptom seen
on alpha-beauty-pilot was that the MTL beauty checkpoint's training-time
``full_eval`` recall@10 stayed in 0.043–0.050 throughout training, yet
the offline ``run_eval.py`` / ``alpha_search.py`` reading on the same
ckpt was 0.0 across every alpha (including ``α=[0,0,0]``, which the
bypass invariant says must equal vanilla beam search). Sports MTL did
not show the gap (train-time 0.215, offline 0.215).

The likely root cause is a corpus_ids / SID-table mismatch between
training time and eval time: at training time, ``train_decoder*.py``
calls ``tokenizer.precompute_corpus_ids(item_dataset)`` once and freezes
the result into ``model.codebooks`` (a registered buffer; see
``train_decoder.py:104-106`` and ``modules/model.py:82``). At eval time,
``run_eval.py`` re-runs ``precompute_corpus_ids`` against the same
``ItemData`` + RQ-VAE — and if anything has drifted (different RQ-VAE
checkpoint, different item ordering, different processed cache, etc.),
the eval-time SID for each item disagrees with the SID the decoder was
actually trained on.

This script answers a single question: do the saved ``_orig_mod.codebooks``
buffer (train-time SIDs) and the freshly recomputed ``cached_ids[:, :n_layers]``
(eval-time SIDs) agree?

Usage::

    PYTHONPATH=. python scripts/diag_corpus_ids.py \\
        --decoder-ckpt path/to/decoder-mtl-beauty/output/model.tar.gz \\
        --rqvae-ckpt   trained_models/rqvae_amazon_beauty/checkpoint_*.pt \\
        --gin-config   configs/decoder_amazon_beauty_mtl.gin \\
        --dataset      beauty

Each ``--*-ckpt`` argument may be either a ``.pt`` file directly, a
directory containing one, or a ``.tar.gz`` archive (extracted to /tmp).

Output (when SIDs match):

    Train-time codebooks: shape=(12101, 3) ...
    Eval-time corpus_ids: shape=(12101, 3) ...
    Row-equality:   12101 / 12101 (100.00%)

Output (when SIDs drift):

    Row-equality:   17 / 12101 (0.14%)
    Per-level disagreement: lvl0=11982  lvl1=12001  lvl2=12089
    Sample mismatched rows (first 10):
      item=  0  train=[ 12,  47,  33]  eval=[ 18, 102, 211]
      ...

A 0.14% match rate is the smoking gun for the §M-zero-metrics row-1
hypothesis: the decoder learned to predict one SID table, the eval
harness is comparing against an entirely different one.

This script does NOT load the decoder model itself, the aux_head, or
``sentence_transformers``. Only the RQ-VAE + ItemData are constructed,
which keeps cold-start under ~30 s on a laptop given a populated
``dataset/amazon/processed/data_*.pt`` cache.
"""
from __future__ import annotations

import argparse
import os
import sys
import tarfile
from pathlib import Path

# Disable network probes preemptively so wandb / huggingface_hub don't
# block at import time (sentence_transformers in particular pings the
# hub even when offline cache is fine).
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def _resolve_ckpt(path: str) -> Path:
    p = Path(path)
    if p.is_file():
        if p.suffix == ".pt":
            return p
        if p.name.endswith(".tar.gz"):
            extract = Path("/tmp") / f"diag_corpus_{p.stem.replace('.', '_')}"
            extract.mkdir(parents=True, exist_ok=True)
            with tarfile.open(p) as tf:
                tf.extractall(extract)
            p = extract
        else:
            return p
    if p.is_dir():
        existing = sorted(p.rglob("*.pt"))
        if not existing:
            for tarball in p.rglob("*.tar.gz"):
                with tarfile.open(tarball) as tf:
                    tf.extractall(p)
                break
            existing = sorted(p.rglob("*.pt"))
        if not existing:
            raise FileNotFoundError(f"No .pt under {path}")
        for cand in existing:
            if "best" in cand.name or cand.name.startswith("checkpoint_"):
                return cand
        return existing[-1]
    raise FileNotFoundError(path)


def _find_codebooks_in_state(state: dict):
    """Return (key, tensor) for the saved corpus_ids buffer.

    ``train_decoder*.py`` saves the model's state_dict directly. When the
    model is wrapped by ``torch.compile``, the registered ``codebooks``
    buffer (set in ``modules/model.py:82``) is exposed as
    ``_orig_mod.codebooks``; without compile it's just ``codebooks``.
    """
    import torch
    candidates = ("_orig_mod.codebooks", "codebooks")
    for key in candidates:
        if key in state and isinstance(state[key], torch.Tensor):
            return key, state[key]
    # Fall back to suffix match in case some other wrapping prefix is in play.
    for key, val in state.items():
        if key.endswith(".codebooks") or key == "codebooks":
            if isinstance(val, torch.Tensor):
                return key, val
    raise KeyError(
        "No codebooks-shaped buffer in checkpoint state_dict — "
        f"got {list(state.keys())[:8]}..."
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--decoder-ckpt", "--decoder_ckpt", required=True,
                   dest="decoder_ckpt",
                   help="Decoder ckpt (.pt / .tar.gz / dir) whose train-time codebooks we want to read.")
    p.add_argument("--rqvae-ckpt", "--rqvae_ckpt", required=True,
                   dest="rqvae_ckpt",
                   help="RQ-VAE ckpt to use when recomputing eval-time SIDs. Must match the one used at training time.")
    p.add_argument("--gin-config", "--gin_config", default=None,
                   dest="gin_config",
                   help="Optional gin config — its train_mtl/train.* bindings drive the architecture lookup.")
    p.add_argument("--dataset", default="beauty",
                   choices=["beauty", "sports", "toys", "steam", "ml1m", "ml32m"])
    p.add_argument("--dataset-folder", "--dataset_folder",
                   default="dataset/amazon", dest="dataset_folder")
    p.add_argument("--dataset-split", "--dataset_split",
                   default=None, dest="dataset_split",
                   help="Defaults to --dataset value; pass-through to ItemData.")
    p.add_argument("--max-print-mismatches", type=int, default=10,
                   dest="max_print_mismatches",
                   help="How many mismatched rows to print verbatim.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    import torch
    import gin

    # Make project modules importable when run from anywhere.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    if args.gin_config:
        gin.parse_config_file(args.gin_config)

    # Register both gin scopes so train_mtl.* and train.* are queryable.
    import train_decoder  # noqa: F401
    import train_decoder_mtl  # noqa: F401
    from data.processed import ItemData, RecDataset
    from evaluate._arch import get_arch
    from modules.tokenizer.semids import SemanticIdTokenizer

    arch = get_arch()
    n_levels = arch["vae_n_layers"]

    # ------------------------------------------------------------------
    # 1. Read train-time codebooks from the decoder checkpoint.
    # ------------------------------------------------------------------
    decoder_pt = _resolve_ckpt(args.decoder_ckpt)
    ckpt = torch.load(decoder_pt, map_location="cpu", weights_only=False)
    model_state = ckpt.get("model", ckpt)
    if not isinstance(model_state, dict):
        raise SystemExit(
            f"Checkpoint {decoder_pt} has no 'model' state_dict (got {type(model_state).__name__})"
        )
    train_key, train_codebooks = _find_codebooks_in_state(model_state)
    train_codebooks = train_codebooks.cpu().long()
    print(f"Decoder ckpt: {decoder_pt}")
    print(f"  iter:               {ckpt.get('iter', '<missing>')}")
    print(f"  codebooks key:      {train_key}")
    print(f"  codebooks shape:    {tuple(train_codebooks.shape)}")
    print(f"  codebooks dtype:    {train_codebooks.dtype}")
    if train_codebooks.ndim != 2 or train_codebooks.shape[1] < n_levels:
        raise SystemExit(
            f"Unexpected codebooks shape {tuple(train_codebooks.shape)}; "
            f"expected (N, >= {n_levels})."
        )
    train_codebooks = train_codebooks[:, :n_levels]

    # ------------------------------------------------------------------
    # 2. Build ItemData + tokenizer + RQ-VAE for eval-time recomputation.
    # ------------------------------------------------------------------
    rqvae_pt = _resolve_ckpt(args.rqvae_ckpt)
    print(f"\nRQ-VAE ckpt: {rqvae_pt}")

    dataset_enum_map = {
        "beauty": RecDataset.AMAZON, "sports": RecDataset.AMAZON,
        "toys": RecDataset.AMAZON, "steam": RecDataset.STEAM,
        "ml1m": RecDataset.ML_1M, "ml32m": RecDataset.ML_32M,
    }
    dataset_enum = dataset_enum_map[args.dataset]
    dataset_split = args.dataset_split or args.dataset

    item_dataset = ItemData(
        root=args.dataset_folder,
        dataset=dataset_enum,
        force_process=False,
        split=dataset_split,
    )
    print(f"ItemData: |corpus|={len(item_dataset)}  split={dataset_split}")

    tokenizer = SemanticIdTokenizer(
        input_dim=arch["vae_input_dim"],
        hidden_dims=arch["vae_hidden_dims"],
        output_dim=arch["vae_embed_dim"],
        codebook_size=arch["vae_codebook_size"],
        n_layers=arch["vae_n_layers"],
        n_cat_feats=arch["vae_n_cat_feats"],
        rqvae_weights_path=str(rqvae_pt),
        rqvae_codebook_normalize=arch["vae_codebook_normalize"],
        rqvae_sim_vq=arch["vae_sim_vq"],
    )
    tokenizer.eval()
    tokenizer.precompute_corpus_ids(item_dataset)
    eval_cached = tokenizer.cached_ids.cpu().long()  # shape (N, n_levels+1)
    eval_codebooks = eval_cached[:, :n_levels]
    print(f"Eval-time cached_ids shape: {tuple(eval_cached.shape)}  "
          f"(extra col is the dedup tag)")

    # ------------------------------------------------------------------
    # 3. Diff.
    # ------------------------------------------------------------------
    if train_codebooks.shape != eval_codebooks.shape:
        print(f"\n!!! shape mismatch: train={tuple(train_codebooks.shape)} "
              f"eval={tuple(eval_codebooks.shape)}")
        print("    can't compare element-wise — corpus size has changed.")
        raise SystemExit(2)

    print(f"\n=== diff: train-time vs eval-time SIDs (shape={tuple(train_codebooks.shape)}) ===")
    n = train_codebooks.shape[0]

    # Per-level disagreement counts.
    per_level = (train_codebooks != eval_codebooks).sum(dim=0).tolist()
    print(f"Per-level disagreement: " +
          "  ".join(f"lvl{i}={c}" for i, c in enumerate(per_level)))

    # Row-level equality (all levels match).
    row_eq = (train_codebooks == eval_codebooks).all(dim=1)
    n_eq = int(row_eq.sum().item())
    pct = 100.0 * n_eq / max(n, 1)
    print(f"Row-equality:    {n_eq} / {n} ({pct:.2f}%)")

    # Prefix-equality at each depth (level 0; levels 0+1; levels 0+1+2).
    for d in range(1, n_levels + 1):
        n_pref = int((train_codebooks[:, :d] == eval_codebooks[:, :d]).all(dim=1).sum().item())
        print(f"  prefix-eq @ depth {d}: {n_pref} / {n} ({100.0 * n_pref / n:.2f}%)")

    # Sample some mismatched rows.
    mismatch_idx = (~row_eq).nonzero(as_tuple=True)[0]
    if mismatch_idx.numel():
        print(f"\nSample mismatched rows (first {min(args.max_print_mismatches, mismatch_idx.numel())}):")
        for i in mismatch_idx[: args.max_print_mismatches].tolist():
            t = train_codebooks[i].tolist()
            e = eval_codebooks[i].tolist()
            print(f"  item={i:>5d}  train={t}  eval={e}")
    else:
        print("\nAll rows match — corpus_ids are byte-identical between train and eval.")

    # Codebook usage histograms (sanity: are both tables exercising the
    # same set of codes? if eval-time is degenerate, level 0 will collapse
    # to a few codes regardless of input).
    K = arch["vae_codebook_size"]
    print(f"\nCodebook usage (unique codes per level / {K}):")
    for d in range(n_levels):
        ut = int(torch.unique(train_codebooks[:, d]).numel())
        ue = int(torch.unique(eval_codebooks[:, d]).numel())
        print(f"  lvl{d}:  train={ut:>4d}   eval={ue:>4d}")

    # Exit code: 0 if perfectly equal, 1 if any mismatch (so a CI step
    # can detect drift if we ever add this as a pre-flight gate).
    raise SystemExit(0 if n_eq == n else 1)


if __name__ == "__main__":
    main()
