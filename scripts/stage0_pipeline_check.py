"""Stage 0 pipeline check — validate the paper's three datasets are ready.

Runs **locally** from the repo root (not on SageMaker). Produces a
`docs/paper_plan_stage0_report.md` summarising:

  1. Whether each upstream RQ-VAE checkpoint is present under
     ``trained_models/`` (sync from upstream if missing).
  2. Whether each RQ-VAE checkpoint can be loaded and ``RqVae.__init__``
     reconstructed from its saved ``model_config`` (or the dataset's gin
     config as fallback).
  3. Whether the dataset's ``ItemData`` pipeline can construct for the
     fork's code path. For ML32M this is the "does the pipeline still work"
     gate called out in ``docs/paper_plan.md`` Stage 0.

Usage::

    python scripts/stage0_pipeline_check.py
    python scripts/stage0_pipeline_check.py --datasets beauty sports ml32m
    python scripts/stage0_pipeline_check.py --skip-data-load   # checkpoint-only

Exit code: 0 if every requested dataset passes its checkpoint + pipeline
check, 1 otherwise. The report is written regardless so the failure mode
is visible.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

# --- Dataset → (gin config, expected checkpoint path) registry ----------

DATASETS = {
    "beauty": {
        "rqvae_ckpt": "trained_models/rqvae_amazon_beauty/checkpoint_high_entropy.pt",
        "rqvae_gin": "configs/rqvae_amazon_beauty.gin",
        "decoder_gin": "configs/decoder_amazon_beauty_mtl.gin",
        "rec_dataset": "AMAZON",
        "dataset_folder": "dataset/amazon",
        "dataset_split": "beauty",
    },
    "sports": {
        "rqvae_ckpt": "trained_models/rqvae_amazon_sports/checkpoint_high_entropy.pt",
        "rqvae_gin": "configs/rqvae_amazon_sports.gin",
        "decoder_gin": "configs/decoder_amazon_sports_mtl.gin",
        "rec_dataset": "AMAZON",
        "dataset_folder": "dataset/amazon",
        "dataset_split": "sports",
    },
    "ml32m": {
        "rqvae_ckpt": "trained_models/rqvae_ml32m/checkpoint_high_entropy.pt",
        "rqvae_gin": "configs/rqvae_ml32m.gin",
        "decoder_gin": "configs/decoder_ml32m_mtl.gin",
        "rec_dataset": "ML_32M",
        "dataset_folder": "dataset/ml-32m",
        "dataset_split": None,
    },
}


def _check_checkpoint_present(spec: dict) -> dict:
    p = Path(spec["rqvae_ckpt"])
    if not p.is_file():
        return {"ok": False, "reason": f"missing: {p}"}
    return {"ok": True, "size_mb": round(p.stat().st_size / 1024 / 1024, 1)}


def _check_checkpoint_loads(spec: dict) -> dict:
    try:
        import torch

        from modules.rqvae import RqVae
    except Exception as e:
        return {"ok": False, "reason": f"import failed: {e}"}

    try:
        state = torch.load(spec["rqvae_ckpt"], map_location="cpu", weights_only=False)
    except Exception as e:
        return {"ok": False, "reason": f"torch.load failed: {e}"}

    model_cfg = state.get("model_config", {}) or {}
    model_cfg = {
        k: v for k, v in model_cfg.items()
        if k in RqVae.__init__.__code__.co_varnames and k not in ("self", "__class__")
    }

    # Fill in any missing kwargs from the gin config so the constructor works.
    try:
        import gin
        gin.clear_config()
        import train_rqvae  # noqa: F401 — registers configurables
        gin.parse_config_file(spec["rqvae_gin"])

        def _q(name, default):
            try:
                return gin.query_parameter(f"train_rqvae.train.{name}")
            except Exception:
                return default

        defaults = dict(
            input_dim=_q("vae_input_dim", 768),
            embed_dim=_q("vae_embed_dim", 32),
            hidden_dims=list(_q("vae_hidden_dims", (512, 256, 128))),
            codebook_size=_q("vae_codebook_size", 256),
            n_layers=_q("vae_n_layers", 3),
            n_cat_features=_q("vae_n_cat_feats", 0),
        )
        for k, v in defaults.items():
            model_cfg.setdefault(k, v)
    except Exception as e:
        return {"ok": False, "reason": f"gin config parse failed: {e}"}

    try:
        rqvae = RqVae(**model_cfg)
        rqvae.load_state_dict(state["model"])
    except Exception as e:
        return {
            "ok": False,
            "reason": f"RqVae reconstruct/load failed: {e}",
            "attempted_model_cfg_keys": sorted(model_cfg.keys()),
        }

    return {
        "ok": True,
        "n_layers": rqvae.n_layers,
        "codebook_size": rqvae.codebook_size,
        "embed_dim": rqvae.embed_dim,
    }


def _check_data_loads(spec: dict) -> dict:
    """Construct ItemData; for ML32M this surfaces any lingering pipeline issues.

    **Fail-fast on missing processed cache.** ``ItemData`` will silently
    invoke ``raw_data.process()`` if the processed cache is absent, and for
    ML32M that pulls a 32M-rating dataset and runs Sentence-T5 on ~86k
    items — a multi-hour CPU job. Stage 0 is meant to be a cheap
    pre-flight; the real data-pipeline gate runs on SageMaker where the
    preprocessed channel is mounted from S3. So this check refuses to
    materialise raw data and returns an actionable error instead.
    """
    folder = Path(spec["dataset_folder"])
    if not folder.exists():
        return {
            "ok": False,
            "reason": (
                f"dataset folder missing: {folder}. Sync from S3 with: "
                f"aws s3 sync $RQVAE_S3_BASE/datasets/{folder.name}/ {folder}/ "
                f"or pass --skip-data-load to defer the data-load gate to SageMaker."
            ),
        }

    processed_dir = folder / "processed"
    processed_pts = sorted(processed_dir.glob("*.pt")) if processed_dir.exists() else []
    if not processed_pts:
        return {
            "ok": False,
            "reason": (
                f"processed cache missing under {processed_dir}/ — refusing to "
                f"trigger raw preprocessing (would take hours on CPU for ML32M). "
                f"Sync with: aws s3 sync $RQVAE_S3_BASE/datasets/{folder.name}/ "
                f"{folder}/  OR re-run with --skip-data-load to defer the "
                f"data-load gate to SageMaker."
            ),
            "processed_dir": str(processed_dir),
        }

    try:
        from data.processed import ItemData, RecDataset
    except Exception as e:
        return {"ok": False, "reason": f"import failed: {e}"}

    rec_dataset = getattr(RecDataset, spec["rec_dataset"], None)
    if rec_dataset is None:
        return {"ok": False, "reason": f"unknown RecDataset enum: {spec['rec_dataset']}"}

    try:
        kwargs = dict(
            root=str(folder),
            dataset=rec_dataset,
            force_process=False,
            train_test_split="all",
        )
        if spec["dataset_split"] is not None:
            kwargs["split"] = spec["dataset_split"]
        items = ItemData(**kwargs)
    except Exception as e:
        return {
            "ok": False,
            "reason": f"ItemData construct failed: {type(e).__name__}: {e}",
            "traceback": traceback.format_exc().splitlines()[-5:],
        }

    try:
        n_items = len(items)
        sample = items[[0]] if n_items else None
        x_shape = tuple(sample.x.shape) if sample is not None else None
    except Exception as e:
        return {"ok": False, "reason": f"ItemData index failed: {e}"}

    return {
        "ok": True,
        "n_items": n_items,
        "sample_x_shape": x_shape,
        "processed_pt": str(processed_pts[0]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    ap.add_argument("--skip-data-load", action="store_true",
                    help="Skip ItemData construction; only checks checkpoints.")
    ap.add_argument("--report-path", default="docs/paper_plan_stage0_report.md")
    ap.add_argument("--json-path", default=None,
                    help="Also write the raw JSON report to this path.")
    args = ap.parse_args()

    report: dict[str, dict] = {}
    all_ok = True
    for ds in args.datasets:
        spec = DATASETS[ds]
        ds_report = {
            "checkpoint_present": _check_checkpoint_present(spec),
        }
        if ds_report["checkpoint_present"]["ok"]:
            ds_report["checkpoint_loads"] = _check_checkpoint_loads(spec)
            if not args.skip_data_load:
                ds_report["data_loads"] = _check_data_loads(spec)
        report[ds] = ds_report

        ds_ok = all(
            step.get("ok", False)
            for step in ds_report.values()
        )
        all_ok = all_ok and ds_ok

    report_md = _render_markdown(report)
    Path(args.report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_path).write_text(report_md)
    print(report_md)
    print(f"\nWrote report: {args.report_path}")

    if args.json_path:
        Path(args.json_path).write_text(json.dumps(report, indent=2, default=str))
        print(f"Wrote JSON:   {args.json_path}")

    sys.exit(0 if all_ok else 1)


def _render_markdown(report: dict) -> str:
    lines = ["# Stage 0 pipeline check report", ""]
    lines.append("| Dataset | Checkpoint | Loads | ItemData | Verdict |")
    lines.append("|---|---|---|---|---|")
    for ds, r in report.items():
        ckpt = "✅" if r.get("checkpoint_present", {}).get("ok") else "❌"
        loads = (
            "✅" if r.get("checkpoint_loads", {}).get("ok")
            else ("❌" if "checkpoint_loads" in r else "—")
        )
        data = (
            "✅" if r.get("data_loads", {}).get("ok")
            else ("❌" if "data_loads" in r else "—")
        )
        verdict = "READY" if all(v.get("ok") for v in r.values()) else "BLOCKED"
        lines.append(f"| {ds} | {ckpt} | {loads} | {data} | {verdict} |")
    lines.append("")

    for ds, r in report.items():
        lines.append(f"## {ds}")
        lines.append("")
        for step, result in r.items():
            status = "ok" if result.get("ok") else "FAIL"
            lines.append(f"- **{step}** — {status}")
            for k, v in result.items():
                if k == "ok":
                    continue
                lines.append(f"    - {k}: `{v}`")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
