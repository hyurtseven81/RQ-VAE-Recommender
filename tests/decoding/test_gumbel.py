"""Tests for GumbelTopKBeamSearch."""
import torch
import pytest

from modules.decoding.gumbel import GumbelTopKBeamSearch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_corpus_and_check_fn(vocab: int, L: int, n_items: int, seed: int = 42):
    rng = torch.Generator()
    rng.manual_seed(seed)
    corpus = torch.randint(0, vocab, (n_items, L), generator=rng)
    corpus = torch.unique(corpus, dim=0)

    def check_valid_prefix(prefix: torch.Tensor) -> torch.Tensor:
        depth = prefix.shape[1]
        trimmed = corpus[:, :depth]
        return (trimmed.unsqueeze(1) == prefix.unsqueeze(0)).all(dim=2).any(dim=0)

    return corpus, check_valid_prefix


def run_gumbel_h0(tau, seed, vocab=8, k=4, B=2, n_cands=6):
    """Run Gumbel expand at h=0 with given seed and tau."""
    _, check_fn = make_corpus_and_check_fn(vocab=vocab, L=2, n_items=20)
    strategy = GumbelTopKBeamSearch(tau=tau)
    torch.manual_seed(seed)
    probas = torch.softmax(torch.randn(B, vocab), dim=-1)
    torch.manual_seed(seed + 100)
    beams, lp, _ = strategy.expand(
        beams=None,
        log_probas=torch.zeros(B, k),
        probas=probas,
        h=0,
        n_cands=n_cands,
        check_valid_fn=check_fn,
        codebook_embs=[],
    )
    return beams, lp


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_seed_reproducibility():
    """Same torch seed must produce identical beams."""
    beams1, lp1 = run_gumbel_h0(tau=1.0, seed=7)
    beams2, lp2 = run_gumbel_h0(tau=1.0, seed=7)
    assert torch.equal(beams1, beams2), "Beams not reproducible with same seed"
    assert torch.allclose(lp1, lp2), "Log-probas not reproducible with same seed"


def test_different_seeds_differ():
    """Different seeds should (with high probability) produce different beams."""
    beams1, _ = run_gumbel_h0(tau=1.0, seed=7)
    beams2, _ = run_gumbel_h0(tau=1.0, seed=99)
    # Not identical (this will hold with overwhelming probability for vocab=8, k=4)
    assert not torch.equal(beams1, beams2), \
        "Different seeds produced identical beams (very unlikely)"


def test_gumbel_returns_k_beams():
    """Returned beam count == k."""
    vocab, k, B, n_cands = 8, 4, 3, 6
    _, check_fn = make_corpus_and_check_fn(vocab=vocab, L=2, n_items=20)
    strategy = GumbelTopKBeamSearch(tau=1.0)
    torch.manual_seed(0)
    probas = torch.softmax(torch.randn(B, vocab), dim=-1)
    beams, lp, parent_idx = strategy.expand(
        beams=None,
        log_probas=torch.zeros(B, k),
        probas=probas,
        h=0,
        n_cands=n_cands,
        check_valid_fn=check_fn,
        codebook_embs=[],
    )
    assert beams.shape == (B, k, 1), f"Expected (B,k,1), got {beams.shape}"
    assert lp.shape == (B, k)
    assert parent_idx.numel() == 0


def test_gumbel_returns_k_beams_h1():
    """Returned beam count == k at h=1."""
    vocab, k, B, n_cands = 8, 4, 2, 6
    L = 2
    _, check_fn = make_corpus_and_check_fn(vocab=vocab, L=L, n_items=20)
    strategy = GumbelTopKBeamSearch(tau=1.0)

    beams_h0 = torch.randint(0, vocab, (B, k, 1))
    lp_h0 = torch.zeros(B, k)

    torch.manual_seed(5)
    probas_h1 = torch.softmax(torch.randn(B * k, vocab), dim=-1)
    beams_h1, lp_h1, parent_idx = strategy.expand(
        beams=beams_h0,
        log_probas=lp_h0,
        probas=probas_h1,
        h=1,
        n_cands=n_cands,
        check_valid_fn=check_fn,
        codebook_embs=[],
    )
    assert beams_h1.shape == (B, k, 2)
    assert lp_h1.shape == (B, k)
    assert parent_idx.shape == (B * k,)


def _mean_pairwise_rank_diff(beams: torch.Tensor, probas: torch.Tensor) -> float:
    """Mean pairwise rank-difference of selected tokens under the probability distribution.

    Low tau: tokens selected are concentrated around highest-probability tokens (low rank diff).
    High tau: tokens selected are more spread out (higher rank diff).
    """
    B, k, _ = beams.shape
    # Get ranks of selected tokens under the original probability distribution
    _, sorted_idx = probas[0].sort(descending=True)
    rank_map = torch.zeros(probas.size(1), dtype=torch.float)
    rank_map[sorted_idx] = torch.arange(probas.size(1), dtype=torch.float)

    selected_tokens = beams[0, :, 0]  # [k]
    ranks = rank_map[selected_tokens]
    return ranks.mean().item()


def test_tau_gt_1_increases_diversity():
    """High tau (10.0) should produce beams with higher average token rank (more spread out)
    than low tau (0.1), measured over many seeds."""
    vocab = 32
    k = 4
    B = 1
    n_cands = 30
    _, check_fn = make_corpus_and_check_fn(vocab=vocab, L=2, n_items=100)

    n_trials = 50
    mean_rank_low_tau = []
    mean_rank_high_tau = []

    for seed in range(n_trials):
        torch.manual_seed(seed)
        probas = torch.softmax(torch.randn(B, vocab), dim=-1)

        for tau, rank_list in [(0.1, mean_rank_low_tau), (10.0, mean_rank_high_tau)]:
            strategy = GumbelTopKBeamSearch(tau=tau)
            torch.manual_seed(seed + 1000)
            beams, _, _ = strategy.expand(
                beams=None,
                log_probas=torch.zeros(B, k),
                probas=probas.clone(),
                h=0,
                n_cands=n_cands,
                check_valid_fn=check_fn,
                codebook_embs=[],
            )
            rank_list.append(_mean_pairwise_rank_diff(beams, probas))

    mean_low = sum(mean_rank_low_tau) / n_trials
    mean_high = sum(mean_rank_high_tau) / n_trials
    assert mean_high > mean_low, (
        f"Expected high-tau beams to have higher average rank (more exploration). "
        f"mean_rank_low_tau={mean_low:.2f}, mean_rank_high_tau={mean_high:.2f}"
    )
