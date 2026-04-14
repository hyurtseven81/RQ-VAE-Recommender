"""Tests for residual entropy analysis module."""
import torch
import numpy as np
import pytest
from unittest.mock import MagicMock


def make_tiny_rqvae(n_levels=3, embed_dim=8, codebook_size=4):
    """Create a mock RQ-VAE that returns known residuals.

    Residual norms decrease with codebook level:
      level 0 -> norm = n_levels
      level 1 -> norm = n_levels - 1
      ...
      level L-1 -> norm = 1
    """
    from types import SimpleNamespace

    def get_semantic_ids(x, gumbel_t=0.001):
        B = x.shape[0]
        # Decreasing residual norms: level 0 > level 1 > level 2
        residuals = torch.stack(
            [torch.ones(embed_dim, B) * (n_levels - l) for l in range(n_levels)]
        )  # [L, D, B]
        sem_ids = torch.randint(0, codebook_size, (n_levels, B))  # [L, B]
        return SimpleNamespace(
            residuals=residuals,
            sem_ids=sem_ids,
            embeddings=torch.zeros(n_levels, embed_dim, B),
        )

    mock = MagicMock()
    mock.get_semantic_ids = get_semantic_ids
    mock.eval = lambda: mock
    return mock


def test_compute_residual_stats_shape():
    """compute_residual_stats returns correct columns."""
    from modules.analysis.residual_entropy import compute_residual_stats

    rqvae = make_tiny_rqvae(n_levels=3, embed_dim=8)
    item_features = torch.randn(12, 16)
    df = compute_residual_stats(rqvae, item_features, batch_size=4)
    assert set(df.columns) >= {"level", "item_id", "residual_norm", "codebook_id"}
    assert len(df) == 12 * 3  # 12 items * 3 levels


def test_residual_norms_decrease():
    """For our mock RQ-VAE, mean residual norm should decrease with level."""
    from modules.analysis.residual_entropy import compute_residual_stats, compute_level_statistics

    rqvae = make_tiny_rqvae(n_levels=3, embed_dim=8)
    item_features = torch.randn(20, 16)
    item_stats = compute_residual_stats(rqvae, item_features, batch_size=10)
    level_stats = compute_level_statistics(item_stats)
    norms = level_stats.sort_values("level")["mean_residual_norm"].values
    assert norms[0] > norms[1] > norms[2], f"Expected decreasing norms, got {norms}"


def test_alpha_entropy_correlation():
    """High correlation when alpha matches entropy pattern."""
    from modules.analysis.residual_entropy import compute_alpha_entropy_correlation
    import pandas as pd

    level_stats = pd.DataFrame(
        {
            "level": [0, 1, 2],
            "empirical_entropy": [8.0, 5.0, 2.0],  # decreasing
        }
    )
    alpha = [0.8, 0.5, 0.2]  # also decreasing
    result = compute_alpha_entropy_correlation(level_stats, alpha)
    assert result["spearman_rho"] > 0.9, f"Expected high correlation, got {result}"


def test_level_statistics_columns():
    """compute_level_statistics returns all required columns."""
    from modules.analysis.residual_entropy import compute_residual_stats, compute_level_statistics

    rqvae = make_tiny_rqvae()
    df = compute_residual_stats(rqvae, torch.randn(8, 16), batch_size=8)
    level_stats = compute_level_statistics(df)
    required = {"level", "mean_residual_norm", "empirical_entropy", "effective_utilization"}
    assert required.issubset(set(level_stats.columns))
