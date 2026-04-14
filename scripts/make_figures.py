"""Generate all paper figures from results/all_runs.parquet and residual entropy parquets.

Usage:
    python scripts/make_figures.py --results results/all_runs.parquet --output paper/figures/
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=1.1)
DATASETS = ["beauty", "sports", "toys", "steam"]
PALETTE = {
    "B0 (Vanilla TIGER)": "#4878cf",
    "B1 (LIGER)": "#d65f5f",
    "H1a (DBS)": "#6acc65",
    "H1b (Gumbel)": "#b47cc7",
    "H1c (Hybrid)": "#c4ad66",
    "H2a (SASRec rerank)": "#77bedb",
    "H2b-grid": "#f28e2b",
    "H2b-learned": "#e15759",
}

CONFIG_LABELS = {
    ("vanilla", "vanilla"): "B0 (Vanilla TIGER)",
    ("liger", "liger"): "B1 (LIGER)",
    ("vanilla", "dbs"): "H1a (DBS)",
    ("vanilla", "gumbel_topk"): "H1b (Gumbel)",
    ("vanilla", "hybrid"): "H1c (Hybrid)",
    ("mtl", "sasrec_rerank"): "H2a (SASRec rerank)",
    ("mtl", "level_aware_mix_grid"): "H2b-grid",
    ("mtl", "level_aware_mix_learned"): "H2b-learned",
}


def make_f2_main_results(df: pd.DataFrame, output_dir: Path):
    """F2: Main results bar chart — Recall@10 × configs × datasets with 95% CIs."""
    fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=False)
    for ax, dataset in zip(axes, DATASETS):
        sub = df[df["dataset"] == dataset].copy()
        sub["config"] = sub.apply(
            lambda r: CONFIG_LABELS.get((r["decoder_type"], r["decoding_strategy"]), "other"), axis=1
        )
        agg = sub.groupby("config")["recall@10"].agg(["mean", "std", "count"]).reset_index()
        agg["ci95"] = 1.96 * agg["std"] / np.sqrt(agg["count"])
        agg = agg.sort_values("mean", ascending=False)
        colors = [PALETTE.get(c, "#888888") for c in agg["config"]]
        ax.barh(agg["config"], agg["mean"], xerr=agg["ci95"], color=colors, alpha=0.85, height=0.6)
        ax.set_title(dataset.capitalize())
        ax.set_xlabel("Recall@10")
    plt.tight_layout()
    fig.savefig(output_dir / "f2_main_results.pdf", bbox_inches="tight", dpi=150)
    fig.savefig(output_dir / "f2_main_results.png", bbox_inches="tight", dpi=150)
    plt.close()
    print("Saved F2")


def make_f3_residual_entropy(output_dir: Path, results_dir: Path):
    """F3: Residual norm and entropy by codebook level."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for dataset in DATASETS:
        path = results_dir / f"residual_entropy_{dataset}.parquet"
        if not path.exists():
            print(f"  Missing: {path}")
            continue
        re = pq.read_table(path).to_pandas()
        axes[0].plot(re["level"], re["mean_residual_norm"], marker="o", label=dataset)
        axes[1].plot(re["level"], re["empirical_entropy"], marker="s", label=dataset)
    axes[0].set_xlabel("Codebook level $l$")
    axes[0].set_ylabel(r"Mean $\|r_l\|_2$")
    axes[0].set_title("Residual norm by level")
    axes[0].legend()
    axes[1].set_xlabel("Codebook level $l$")
    axes[1].set_ylabel(r"$H(c_l \mid c_{<l})$")
    axes[1].set_title("Codebook entropy by level")
    axes[1].legend()
    plt.tight_layout()
    fig.savefig(output_dir / "f3_residual_entropy.pdf", bbox_inches="tight", dpi=150)
    fig.savefig(output_dir / "f3_residual_entropy.png", bbox_inches="tight", dpi=150)
    plt.close()
    print("Saved F3")


def make_f4_alpha_schedule(df: pd.DataFrame, output_dir: Path):
    """F4: Alpha schedule (learned + grid) vs. codebook level, per dataset."""
    mixing_configs = df[df["decoding_strategy"].isin(["level_aware_mix_grid", "level_aware_mix_learned"])]
    if mixing_configs.empty:
        print("  No level-aware mixing results yet, skipping F4")
        return
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    for ax, dataset in zip(axes, DATASETS):
        sub = mixing_configs[mixing_configs["dataset"] == dataset]
        for strategy, label, marker in [
            ("level_aware_mix_grid", "Grid α", "o"),
            ("level_aware_mix_learned", "Learned α", "s"),
        ]:
            rows = sub[sub["decoding_strategy"] == strategy]
            if rows.empty:
                continue
            # Best run by recall@10
            best = rows.sort_values("recall@10", ascending=False).iloc[0]
            alpha = json.loads(best["alpha_schedule"])
            ax.plot(range(len(alpha)), alpha, marker=marker, label=label)
        ax.set_title(dataset.capitalize())
        ax.set_xlabel("Codebook level $l$")
        ax.set_ylabel(r"$\alpha_l$")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(output_dir / "f4_alpha_schedule.pdf", bbox_inches="tight", dpi=150)
    fig.savefig(output_dir / "f4_alpha_schedule.png", bbox_inches="tight", dpi=150)
    plt.close()
    print("Saved F4")


def make_f7_beam_sweep(df: pd.DataFrame, output_dir: Path):
    """F7: Beam size sensitivity (appendix)."""
    best_config = df[df["decoding_strategy"] == "level_aware_mix_learned"]
    if best_config.empty or "beam_size" not in df.columns:
        print("  No beam sweep data yet, skipping F7")
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    for dataset in DATASETS:
        sub = best_config[best_config["dataset"] == dataset]
        agg = sub.groupby("beam_size")["recall@10"].mean().reset_index()
        ax.plot(agg["beam_size"], agg["recall@10"], marker="o", label=dataset)
    ax.set_xlabel("Beam size $k$")
    ax.set_ylabel("Recall@10")
    ax.set_title("Beam size sensitivity")
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_dir / "f7_beam_sweep.pdf", bbox_inches="tight", dpi=150)
    fig.savefig(output_dir / "f7_beam_sweep.png", bbox_inches="tight", dpi=150)
    plt.close()
    print("Saved F7")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/all_runs.parquet")
    parser.add_argument("--residual-dir", default="results")
    parser.add_argument("--output", default="paper/figures")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame()
    if Path(args.results).exists():
        df = pq.read_table(args.results).to_pandas()
        print(f"Loaded {len(df)} result rows")
    else:
        print(f"Warning: {args.results} not found, skipping data-dependent figures")

    make_f3_residual_entropy(output_dir, Path(args.residual_dir))
    if not df.empty:
        make_f2_main_results(df, output_dir)
        make_f4_alpha_schedule(df, output_dir)
        make_f7_beam_sweep(df, output_dir)
    print("Done. Figures saved to", output_dir)


if __name__ == "__main__":
    main()
