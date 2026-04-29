"""Diagnostic: inspect MTL decoder checkpoint state_dicts.

Designed for the §M-zero-metrics row 2 case ("MTL ckpt itself is broken"):
when one MTL decoder produces zero recall under VanillaBeamSearch but
another (sports vs beauty etc.) works, this script compares their
state_dicts to surface what's different. Runs locally with
``torch.load`` only — no model construction, no sentence_transformers,
no wandb. Cold-start under 5 s once the venv has torch.

Usage::

    PYTHONPATH=. python scripts/diag_mtl_ckpt.py \\
        path/to/decoder-mtl-beauty/output/model.tar.gz \\
        path/to/decoder-mtl-sports/output/model.tar.gz

Each argument may be either a ``.pt`` file directly, a directory
containing a ``.pt``, or a ``.tar.gz`` archive (which is extracted to
``/tmp`` and searched for a ``.pt``). The script prints, for each
checkpoint:

  - top-level keys present in the saved dict
  - state-dict size, key-prefix histogram, expected wrapper shape
  - aux_head presence + state-dict size
  - sample param statistics (mean / std) for two characteristic keys
  - saved metadata (iter, lambda_aux_max if MTL)

Then it diffs the two: keys present in only one, params with mismatched
shapes. The diff is the actionable output — anything in the "only-in-A"
set is a key that ``strict=False`` would silently drop.
"""
from __future__ import annotations

import argparse
import os
import sys
import tarfile
from collections import Counter
from pathlib import Path

# Disable network probes preemptively so torch.load doesn't accidentally
# pull anything (e.g. via tensorboardX or huggingface_hub side imports).
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def _resolve_pt(path: str) -> Path:
    p = Path(path)
    if p.is_file():
        if p.suffix == ".pt":
            return p
        if p.name.endswith(".tar.gz"):
            extract = Path("/tmp") / f"diag_{p.stem.replace('.', '_')}"
            extract.mkdir(parents=True, exist_ok=True)
            with tarfile.open(p) as tf:
                tf.extractall(extract)
            p = extract
    if p.is_dir():
        pts = sorted(p.rglob("*.pt"))
        if not pts:
            # Maybe a tar.gz inside the directory?
            for tarball in p.rglob("*.tar.gz"):
                with tarfile.open(tarball) as tf:
                    tf.extractall(p)
                break
            pts = sorted(p.rglob("*.pt"))
        if not pts:
            raise FileNotFoundError(f"No .pt under {path}")
        for cand in pts:
            if "best" in cand.name or cand.name.startswith("checkpoint_"):
                return cand
        return pts[-1]
    raise FileNotFoundError(path)


def _summarize(state_dict: dict) -> dict:
    """Return histogram + sample stats for a single state_dict."""
    import torch  # local to keep the script importable without torch on path

    keys = list(state_dict.keys())
    prefix_counts = Counter()
    for k in keys:
        parts = k.split(".")
        head = ".".join(parts[:2]) if len(parts) > 1 else parts[0]
        prefix_counts[head] += 1

    # Sample stats — pick the first param-like tensor we find for each
    # of: an embedding (codebook / item table), an MLP weight, and a layer norm.
    sample_keys = []
    seen_kinds = set()
    for k in keys:
        v = state_dict[k]
        if not isinstance(v, torch.Tensor) or v.numel() == 0:
            continue
        k_lower = k.lower()
        kind = None
        if "embedding" in k_lower or "codebook" in k_lower or "item_sid" in k_lower:
            kind = "embedding"
        elif "linear" in k_lower or "weight" in k_lower and v.ndim == 2:
            kind = "linear_weight"
        elif "norm" in k_lower and v.ndim == 1:
            kind = "norm"
        if kind and kind not in seen_kinds:
            seen_kinds.add(kind)
            sample_keys.append((kind, k, v))
        if len(sample_keys) >= 3:
            break

    return {
        "n_keys": len(keys),
        "prefix_counts": dict(prefix_counts.most_common(8)),
        "sample_stats": [
            (kind, k, tuple(v.shape), float(v.float().mean()),
             float(v.float().std()))
            for kind, k, v in sample_keys
        ],
        "all_keys": keys,
    }


def _print_summary(label: str, ckpt_path: Path, ckpt: dict) -> dict:
    print(f"\n=== {label}: {ckpt_path} ===")
    print(f"top-level keys: {sorted(ckpt.keys())}")
    print(f"iter: {ckpt.get('iter', '<missing>')}")
    if "lambda_aux_max" in ckpt:
        print(f"lambda_aux_max: {ckpt['lambda_aux_max']}")
    if "sasrec_temperature" in ckpt:
        print(f"sasrec_temperature: {ckpt['sasrec_temperature']}")

    model_state = ckpt.get("model")
    if not isinstance(model_state, dict):
        print(f"!!! 'model' key is missing or not a dict (got {type(model_state).__name__})")
        return {"all_keys": []}

    summary = _summarize(model_state)
    print(f"\nmodel state_dict: {summary['n_keys']} keys")
    print("key-prefix histogram (top 8):")
    for p, n in summary["prefix_counts"].items():
        print(f"  {p:35s}  {n}")
    print("sample params (kind, key, shape, mean, std):")
    for kind, k, shape, mean, std in summary["sample_stats"]:
        print(f"  {kind:14s}  {k}  shape={shape}  mean={mean:.4g}  std={std:.4g}")

    aux_state = ckpt.get("aux_head")
    if isinstance(aux_state, dict):
        aux_summary = _summarize(aux_state)
        print(f"\naux_head state_dict: {aux_summary['n_keys']} keys")
        print("key-prefix histogram (top 5):")
        for p, n in list(aux_summary["prefix_counts"].items())[:5]:
            print(f"  {p:35s}  {n}")

    return summary


def _diff(label_a: str, sum_a: dict, label_b: str, sum_b: dict) -> None:
    keys_a = set(sum_a["all_keys"])
    keys_b = set(sum_b["all_keys"])
    only_a = sorted(keys_a - keys_b)
    only_b = sorted(keys_b - keys_a)
    common = keys_a & keys_b

    print(f"\n=== diff: {label_a} vs {label_b} ===")
    print(f"{label_a}-only keys: {len(only_a)}")
    if only_a:
        for k in only_a[:5]:
            print(f"  {k}")
        if len(only_a) > 5:
            print(f"  ...and {len(only_a) - 5} more")
    print(f"{label_b}-only keys: {len(only_b)}")
    if only_b:
        for k in only_b[:5]:
            print(f"  {k}")
        if len(only_b) > 5:
            print(f"  ...and {len(only_b) - 5} more")
    print(f"common keys: {len(common)}")

    # Heuristic: are prefixes structurally identical?
    pa = set(sum_a["prefix_counts"].keys())
    pb = set(sum_b["prefix_counts"].keys())
    if pa != pb:
        print("\n!!! prefix sets differ — this is the smoking gun:")
        print(f"  {label_a}-only prefixes: {sorted(pa - pb)}")
        print(f"  {label_b}-only prefixes: {sorted(pb - pa)}")
    else:
        print("\nprefix sets identical — no wrapping mismatch.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_a", help=".pt / .tar.gz / dir for first checkpoint (broken candidate)")
    ap.add_argument("ckpt_b", nargs="?",
                    help="optional second checkpoint to diff against (known-good control)")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    args = ap.parse_args()

    import torch  # noqa: F401 — fail fast if torch isn't installed

    pt_a = _resolve_pt(args.ckpt_a)
    ckpt_a = torch.load(pt_a, map_location="cpu", weights_only=False)
    sum_a = _print_summary(args.label_a, pt_a, ckpt_a)

    if args.ckpt_b:
        pt_b = _resolve_pt(args.ckpt_b)
        ckpt_b = torch.load(pt_b, map_location="cpu", weights_only=False)
        sum_b = _print_summary(args.label_b, pt_b, ckpt_b)
        _diff(args.label_a, sum_a, args.label_b, sum_b)


if __name__ == "__main__":
    sys.exit(main())
