"""Generate LaTeX table code from results/all_runs.parquet.

Produces:
- T1: Main results table (Recall@{5,10,20}, NDCG@{5,10,20} for all configs x datasets)
- T2: Ablation table (Beauty + Sports)
- T3: Per-segment results (Beauty)

Usage:
    python scripts/make_tables.py --results results/all_runs.parquet --output paper/tables/
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

DATASETS = ["beauty", "sports", "toys", "steam"]
METRICS = ["recall@5", "recall@10", "recall@20", "ndcg@5", "ndcg@10", "ndcg@20"]

CONFIG_ORDER = [
    ("vanilla", "vanilla", "B0 (Vanilla TIGER)"),
    ("liger", "liger", "B1 (LIGER)$^\\dagger$"),
    ("vanilla", "dbs", "H1a (DBS-emb)"),
    ("vanilla", "gumbel_topk", "H1b (Gumbel)"),
    ("vanilla", "hybrid", "H1c (Hybrid)"),
    ("mtl", "sasrec_rerank", "H2a (SASRec rerank)"),
    ("mtl", "level_aware_mix_grid", "H2b-grid"),
    ("mtl", "level_aware_mix_learned", "H2b-learned"),
]


def format_cell(mean: float, ci: float, is_best: bool = False) -> str:
    """Format mean +/- CI for LaTeX. Bold if best."""
    s = f"{mean:.4f}$\\pm${ci:.4f}"
    return f"\\textbf{{{s}}}" if is_best else s


def _agg_metric(df: pd.DataFrame, metric: str) -> tuple:
    """Return (mean, 95% CI half-width) for a metric column across seeds."""
    if metric not in df.columns or df.empty:
        return float("nan"), float("nan")
    vals = df[metric].dropna().values
    if len(vals) == 0:
        return float("nan"), float("nan")
    mean = float(np.mean(vals))
    ci = float(1.96 * np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
    return mean, ci


def make_t1_main_results(df: pd.DataFrame, output_dir: Path) -> None:
    """Generate T1: main results table across all datasets and configs."""
    selected_metrics = ["recall@10", "recall@20", "ndcg@10", "ndcg@20"]

    lines = []
    lines.append("% T1: Main Results")
    n_metric_cols = len(selected_metrics) * len(DATASETS)
    lines.append(f"\\begin{{tabular}}{{l{'c' * n_metric_cols}}}")
    lines.append("\\toprule")

    # Header row 1: dataset names spanning metric columns
    header1 = "Method"
    for ds in DATASETS:
        header1 += f" & \\multicolumn{{{len(selected_metrics)}}}{{c}}{{{ds.capitalize()}}}"
    lines.append(header1 + " \\\\")

    # Cmidrule for each dataset group
    cmidrules = []
    for i, _ in enumerate(DATASETS):
        start = 2 + i * len(selected_metrics)
        end = start + len(selected_metrics) - 1
        cmidrules.append(f"\\cmidrule(lr){{{start}-{end}}}")
    lines.append(" ".join(cmidrules))

    # Header row 2: metric names
    header2 = ""
    for _ in DATASETS:
        for m in selected_metrics:
            header2 += f" & {m}"
    lines.append(header2 + " \\\\")
    lines.append("\\midrule")

    # Find best values per (dataset, metric) for bolding
    best: dict = {}
    for ds in DATASETS:
        for m in selected_metrics:
            sub = df[df["dataset"] == ds]
            if m in sub.columns and not sub.empty:
                best[(ds, m)] = sub[m].max()

    # Data rows
    for decoder_type, decoding_strategy, label in CONFIG_ORDER:
        row = label
        for ds in DATASETS:
            sub = df[
                (df["dataset"] == ds)
                & (df["decoder_type"] == decoder_type)
                & (df["decoding_strategy"] == decoding_strategy)
            ]
            for m in selected_metrics:
                mean, ci = _agg_metric(sub, m)
                if np.isnan(mean):
                    row += " & --"
                else:
                    is_best = abs(mean - best.get((ds, m), float("nan"))) < 1e-6
                    row += f" & {format_cell(mean, ci, is_best)}"
        lines.append(row + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    out_file = output_dir / "t1_main_results.tex"
    out_file.write_text("\n".join(lines))
    print(f"Wrote {out_file}")


def make_t2_ablations(df: pd.DataFrame, output_dir: Path) -> None:
    """Generate T2: ablation table on Beauty and Sports."""
    ablation_datasets = ["beauty", "sports"]
    selected_metrics = ["recall@10", "ndcg@10"]

    lines = []
    lines.append("% T2: Ablation Results (Beauty + Sports)")
    n_cols = len(selected_metrics) * len(ablation_datasets)
    lines.append(f"\\begin{{tabular}}{{l{'c' * n_cols}}}")
    lines.append("\\toprule")

    header1 = "Method"
    for ds in ablation_datasets:
        header1 += f" & \\multicolumn{{{len(selected_metrics)}}}{{c}}{{{ds.capitalize()}}}"
    lines.append(header1 + " \\\\")

    cmidrules = []
    for i, _ in enumerate(ablation_datasets):
        start = 2 + i * len(selected_metrics)
        end = start + len(selected_metrics) - 1
        cmidrules.append(f"\\cmidrule(lr){{{start}-{end}}}")
    lines.append(" ".join(cmidrules))

    header2 = ""
    for _ in ablation_datasets:
        for m in selected_metrics:
            header2 += f" & {m}"
    lines.append(header2 + " \\\\")
    lines.append("\\midrule")

    best: dict = {}
    for ds in ablation_datasets:
        for m in selected_metrics:
            sub = df[df["dataset"] == ds]
            if m in sub.columns and not sub.empty:
                best[(ds, m)] = sub[m].max()

    for decoder_type, decoding_strategy, label in CONFIG_ORDER:
        row = label
        for ds in ablation_datasets:
            sub = df[
                (df["dataset"] == ds)
                & (df["decoder_type"] == decoder_type)
                & (df["decoding_strategy"] == decoding_strategy)
            ]
            for m in selected_metrics:
                mean, ci = _agg_metric(sub, m)
                if np.isnan(mean):
                    row += " & --"
                else:
                    is_best = abs(mean - best.get((ds, m), float("nan"))) < 1e-6
                    row += f" & {format_cell(mean, ci, is_best)}"
        lines.append(row + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    out_file = output_dir / "t2_ablations.tex"
    out_file.write_text("\n".join(lines))
    print(f"Wrote {out_file}")


def make_t3_segments(df: pd.DataFrame, output_dir: Path) -> None:
    """Generate T3: per-segment results for Beauty dataset."""
    ds = "beauty"
    segment_col = "segment"
    selected_metrics = ["recall@10", "ndcg@10"]

    beauty_df = df[df["dataset"] == ds].copy()

    if segment_col not in beauty_df.columns or beauty_df.empty:
        print("T3: No segment column found in Beauty results; skipping.")
        # Write a placeholder
        out_file = output_dir / "t3_segments.tex"
        out_file.write_text("% T3: Per-segment results not available (no segment column)\n")
        print(f"Wrote placeholder {out_file}")
        return

    segments = sorted(beauty_df[segment_col].dropna().unique())

    lines = []
    lines.append("% T3: Per-Segment Results (Beauty)")
    n_cols = len(selected_metrics) * len(segments)
    lines.append(f"\\begin{{tabular}}{{l{'c' * n_cols}}}")
    lines.append("\\toprule")

    header1 = "Method"
    for seg in segments:
        header1 += f" & \\multicolumn{{{len(selected_metrics)}}}{{c}}{{Segment {seg}}}"
    lines.append(header1 + " \\\\")

    cmidrules = []
    for i, _ in enumerate(segments):
        start = 2 + i * len(selected_metrics)
        end = start + len(selected_metrics) - 1
        cmidrules.append(f"\\cmidrule(lr){{{start}-{end}}}")
    lines.append(" ".join(cmidrules))

    header2 = ""
    for _ in segments:
        for m in selected_metrics:
            header2 += f" & {m}"
    lines.append(header2 + " \\\\")
    lines.append("\\midrule")

    best: dict = {}
    for seg in segments:
        for m in selected_metrics:
            sub = beauty_df[beauty_df[segment_col] == seg]
            if m in sub.columns and not sub.empty:
                best[(seg, m)] = sub[m].max()

    for decoder_type, decoding_strategy, label in CONFIG_ORDER:
        row = label
        for seg in segments:
            sub = beauty_df[
                (beauty_df[segment_col] == seg)
                & (beauty_df["decoder_type"] == decoder_type)
                & (beauty_df["decoding_strategy"] == decoding_strategy)
            ]
            for m in selected_metrics:
                mean, ci = _agg_metric(sub, m)
                if np.isnan(mean):
                    row += " & --"
                else:
                    is_best = abs(mean - best.get((seg, m), float("nan"))) < 1e-6
                    row += f" & {format_cell(mean, ci, is_best)}"
        lines.append(row + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    out_file = output_dir / "t3_segments.tex"
    out_file.write_text("\n".join(lines))
    print(f"Wrote {out_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/all_runs.parquet")
    parser.add_argument(
        "--liger-baseline", default="results/baselines/liger_paper_numbers.json"
    )
    parser.add_argument("--output", default="paper/tables")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame()
    if Path(args.results).exists():
        df = pq.read_table(args.results).to_pandas()

    # Load LIGER paper numbers and add as rows
    if Path(args.liger_baseline).exists():
        with open(args.liger_baseline) as f:
            liger = json.load(f)
        liger_rows = []
        for dataset in DATASETS:
            if dataset in liger:
                row = {
                    "dataset": dataset,
                    "decoder_type": "liger",
                    "decoding_strategy": "liger",
                    "seed": -1,
                }
                row.update(liger[dataset])
                liger_rows.append(row)
        if liger_rows:
            df = pd.concat([df, pd.DataFrame(liger_rows)], ignore_index=True)

    if df.empty:
        print("No results found. Tables will be generated once experiments complete.")
        return

    make_t1_main_results(df, output_dir)
    make_t2_ablations(df, output_dir)
    make_t3_segments(df, output_dir)
    print(f"Tables saved to {output_dir}")


if __name__ == "__main__":
    main()
