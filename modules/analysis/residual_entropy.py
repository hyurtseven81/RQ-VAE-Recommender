"""Residual entropy analysis for theoretical motivation of per-level alpha mixing.

Computes per-level statistics that motivate the monotone-decreasing alpha schedule:
1. Mean L2 norm of residuals at each codebook level: E[||r_l||_2]
2. Empirical entropy H(c_l | c_{<l}) of codebook assignments
3. Effective codebook utilization (count unique codes used at level l)

These statistics correlate with the learned alpha schedule in the paper (Figure F3, F5).
If Spearman rho(residual_entropy_l, alpha_l) > 0.8, that is the paper's theoretical payoff.

Note on RqVaeOutput shapes (from modules/rqvae.py):
  residuals: [L, D, B]  — residuals[l, :, i] is residual vector at level l for item i
  sem_ids:   [L, B]     — sem_ids[l, i] is the codebook index at level l for item i
"""
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from scipy.stats import spearmanr


def compute_residual_stats(
    rqvae: torch.nn.Module,
    item_features: torch.Tensor,  # [num_items, feature_dim]
    batch_size: int = 256,
    device: str = "cpu",
) -> pd.DataFrame:
    """Compute per-level residual statistics for all items.

    Returns a DataFrame with columns:
        level, item_id, residual_norm, codebook_id
    """
    rqvae.eval()
    records = []
    with torch.no_grad():
        for start in range(0, len(item_features), batch_size):
            batch = item_features[start : start + batch_size].to(device)
            output = rqvae.get_semantic_ids(batch)
            # output.residuals shape: [L, D, B]
            # output.sem_ids shape:   [L, B]
            residuals = output.residuals  # [L, D, B]
            sem_ids = output.sem_ids      # [L, B]
            L, D, B = residuals.shape
            for l in range(L):
                norms = residuals[l].norm(dim=0).cpu().numpy()  # [B]
                if sem_ids.ndim == 2:
                    # Shape is [L, B]
                    ids = sem_ids[l].cpu().numpy()
                else:
                    # Shape is [B, L]
                    ids = sem_ids[:, l].cpu().numpy()
                for i in range(B):
                    records.append(
                        {
                            "level": l,
                            "item_id": start + i,
                            "residual_norm": float(norms[i]),
                            "codebook_id": int(ids[i]),
                        }
                    )
    return pd.DataFrame(records)


def compute_level_statistics(item_stats: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-item stats into per-level statistics.

    Returns DataFrame with columns:
        level, mean_residual_norm, std_residual_norm,
        empirical_entropy, effective_utilization (count unique codes)
    """
    rows = []
    for level in sorted(item_stats["level"].unique()):
        sub = item_stats[item_stats["level"] == level]
        norms = sub["residual_norm"].values

        # Codebook utilization
        code_counts = sub["codebook_id"].value_counts()
        unique_codes = len(code_counts)

        # Empirical entropy H(c_l) from marginal distribution
        probs = code_counts.values / code_counts.values.sum()
        entropy = float(-np.sum(probs * np.log2(probs + 1e-10)))

        rows.append(
            {
                "level": level,
                "mean_residual_norm": float(norms.mean()),
                "std_residual_norm": float(norms.std()),
                "empirical_entropy": entropy,
                "effective_utilization": unique_codes,
            }
        )
    return pd.DataFrame(rows)


def compute_conditional_entropy(
    item_stats: pd.DataFrame,
    n_levels: int,
) -> List[float]:
    """Compute H(c_l | c_{<l}) for each level using chain rule.

    H(c_l | c_{<l}) = H(c_0, c_1, ..., c_l) - H(c_0, ..., c_{l-1})
    Estimated from empirical joint counts.
    """
    # Build wide format: one row per item, one column per level
    wide = item_stats.pivot(index="item_id", columns="level", values="codebook_id").dropna()

    conditional_entropies = []
    for l in range(n_levels):
        # H(c_0, ..., c_l)
        prefix_cols = list(range(l + 1))
        joint_counts = wide[prefix_cols].apply(tuple, axis=1).value_counts()
        joint_probs = joint_counts.values / joint_counts.values.sum()
        h_joint = float(-np.sum(joint_probs * np.log2(joint_probs + 1e-10)))

        if l == 0:
            h_prev = 0.0
        else:
            prev_cols = list(range(l))
            prev_counts = wide[prev_cols].apply(tuple, axis=1).value_counts()
            prev_probs = prev_counts.values / prev_counts.values.sum()
            h_prev = float(-np.sum(prev_probs * np.log2(prev_probs + 1e-10)))

        conditional_entropies.append(h_joint - h_prev)

    return conditional_entropies


def compute_alpha_entropy_correlation(
    level_stats: pd.DataFrame,
    alpha_schedule: List[float],
) -> Dict:
    """Compute Spearman correlation between residual entropy and alpha schedule.

    Returns dict with 'spearman_rho', 'p_value', 'interpretation'.
    A rho > 0.8 is the paper's theoretical payoff.
    """
    entropy_vals = level_stats.sort_values("level")["empirical_entropy"].values
    alpha_vals = np.array(alpha_schedule)

    if len(entropy_vals) != len(alpha_vals):
        return {"error": "length mismatch"}

    rho, p = spearmanr(entropy_vals, alpha_vals)
    return {
        "spearman_rho": float(rho),
        "p_value": float(p),
        "interpretation": (
            "Strong correlation (>0.8): confirms coarse-to-fine mixing principle"
            if rho > 0.8
            else "Weak correlation: prior not confirmed on this dataset"
        ),
    }


def run_analysis(
    rqvae: torch.nn.Module,
    item_features: torch.Tensor,
    output_path: str,
    alpha_schedule: Optional[List[float]] = None,
    batch_size: int = 256,
    device: str = "cpu",
) -> pd.DataFrame:
    """Full pipeline: compute stats, save parquet, optionally compute correlation."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    print("Computing per-item residual statistics...")
    item_stats = compute_residual_stats(rqvae, item_features, batch_size, device)
    level_stats = compute_level_statistics(item_stats)

    n_levels = item_stats["level"].nunique()
    cond_entropies = compute_conditional_entropy(item_stats, n_levels)
    level_stats["conditional_entropy"] = cond_entropies

    pq.write_table(pa.Table.from_pandas(level_stats), output_path)
    print(f"Saved to {output_path}")
    print(level_stats.to_string())

    if alpha_schedule is not None:
        corr = compute_alpha_entropy_correlation(level_stats, alpha_schedule)
        print(f"Alpha-entropy correlation: {corr}")

    return level_stats
