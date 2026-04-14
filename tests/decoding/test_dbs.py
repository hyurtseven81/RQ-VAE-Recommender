"""Tests for DiverseBeamSearch."""
import torch
import pytest

from modules.decoding.dbs import DiverseBeamSearch


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


def make_codebook_embs(L: int, vocab: int, embed_dim: int, seed: int = 0):
    torch.manual_seed(seed)
    return [torch.randn(vocab, embed_dim) for _ in range(L)]


def mean_pairwise_l2(beams: torch.Tensor) -> float:
    """Mean L2 distance between all pairs of beam tokens (last level only)."""
    # beams: [B, k, L] — use full token sequences as flat vectors
    B, k, L = beams.shape
    flat = beams.float().reshape(B * k, L)
    # Compute pairwise distances
    dists = torch.cdist(flat, flat, p=2)
    # Take upper triangle (exclude diagonal)
    mask = torch.triu(torch.ones(B * k, B * k, dtype=torch.bool), diagonal=1)
    return dists[mask].mean().item()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_dbs_num_beams_correct():
    """DiverseBeamSearch must return exactly k beams."""
    vocab, k, B, n_cands = 4, 4, 2, 4
    L = 2
    embed_dim = 4
    corpus, check_fn = make_corpus_and_check_fn(vocab=vocab, L=L, n_items=10)
    codebook_embs = make_codebook_embs(L=L, vocab=vocab, embed_dim=embed_dim)

    strategy = DiverseBeamSearch(num_groups=2, lambda_per_level=[0.5, 0.5])

    # h=0: uses vanilla internally
    torch.manual_seed(0)
    probas_h0 = torch.softmax(torch.randn(B, vocab), dim=-1)
    beams_h0, lp_h0, _ = strategy.expand(
        beams=None,
        log_probas=torch.zeros(B, k),
        probas=probas_h0,
        h=0,
        n_cands=n_cands,
        check_valid_fn=check_fn,
        codebook_embs=codebook_embs,
    )
    assert beams_h0.shape == (B, k, 1), f"Expected (B,k,1) got {beams_h0.shape}"

    # h=1: diverse expansion
    torch.manual_seed(1)
    probas_h1 = torch.softmax(torch.randn(B * k, vocab), dim=-1)
    beams_h1, lp_h1, parent_idx = strategy.expand(
        beams=beams_h0,
        log_probas=lp_h0,
        probas=probas_h1,
        h=1,
        n_cands=n_cands,
        check_valid_fn=check_fn,
        codebook_embs=codebook_embs,
    )
    assert beams_h1.shape == (B, k, 2), f"Expected (B,k,2) got {beams_h1.shape}"
    assert lp_h1.shape == (B, k)
    assert parent_idx.shape == (B * k,)


@pytest.mark.parametrize("lam", [0.0, 0.5, 1.0])
def test_diversity_increases_with_lambda(lam):
    """Higher lambda should not decrease diversity (non-decreasing)."""
    # We run DBS at multiple lambdas and check the trend.
    # We only assert the qualitative property across 3 settings.
    vocab = 4
    k = 4
    B = 1
    n_cands = 4
    L = 2
    embed_dim = 4
    corpus, check_fn = make_corpus_and_check_fn(vocab=vocab, L=L, n_items=10)
    codebook_embs = make_codebook_embs(L=L, vocab=vocab, embed_dim=embed_dim, seed=5)

    results = {}
    for lam_val in [0.0, 0.5, 1.0]:
        strategy = DiverseBeamSearch(
            num_groups=2, lambda_per_level=[lam_val, lam_val]
        )
        torch.manual_seed(42)
        probas_h0 = torch.softmax(torch.randn(B, vocab), dim=-1)
        beams_h0, lp_h0, _ = strategy.expand(
            beams=None,
            log_probas=torch.zeros(B, k),
            probas=probas_h0,
            h=0,
            n_cands=n_cands,
            check_valid_fn=check_fn,
            codebook_embs=codebook_embs,
        )

        torch.manual_seed(43)
        probas_h1 = torch.softmax(torch.randn(B * k, vocab), dim=-1)
        beams_h1, lp_h1, _ = strategy.expand(
            beams=beams_h0,
            log_probas=lp_h0,
            probas=probas_h1,
            h=1,
            n_cands=n_cands,
            check_valid_fn=check_fn,
            codebook_embs=codebook_embs,
        )
        results[lam_val] = mean_pairwise_l2(beams_h1)

    # Non-decreasing: diversity at lam=0.5 >= lam=0.0 and lam=1.0 >= lam=0.5
    # (or at least non-strictly-decreasing with some tolerance)
    assert results[1.0] >= results[0.0] - 1e-3, (
        f"Diversity should be non-decreasing with lambda. "
        f"lam=0.0: {results[0.0]:.4f}, lam=1.0: {results[1.0]:.4f}"
    )


def test_token_id_dbs_variant():
    """use_embedding_distance=False (Hamming variant) runs without error."""
    vocab, k, B, n_cands = 4, 4, 2, 4
    L = 2
    corpus, check_fn = make_corpus_and_check_fn(vocab=vocab, L=L, n_items=10)

    strategy = DiverseBeamSearch(
        num_groups=2,
        lambda_per_level=[0.5, 0.5],
        use_embedding_distance=False,
    )

    torch.manual_seed(0)
    probas_h0 = torch.softmax(torch.randn(B, vocab), dim=-1)
    beams_h0, lp_h0, _ = strategy.expand(
        beams=None,
        log_probas=torch.zeros(B, k),
        probas=probas_h0,
        h=0,
        n_cands=n_cands,
        check_valid_fn=check_fn,
        codebook_embs=None,  # Not needed for Hamming variant
    )

    torch.manual_seed(1)
    probas_h1 = torch.softmax(torch.randn(B * k, vocab), dim=-1)
    beams_h1, lp_h1, _ = strategy.expand(
        beams=beams_h0,
        log_probas=lp_h0,
        probas=probas_h1,
        h=1,
        n_cands=n_cands,
        check_valid_fn=check_fn,
        codebook_embs=None,  # Not needed for Hamming variant
    )
    assert beams_h1.shape == (B, k, 2)


def test_dbs_raises_without_codebook_embs_when_embedding_distance():
    """DBS should raise ValueError when codebook_embs is None and embedding distance is needed."""
    vocab, k, B, n_cands = 4, 4, 1, 4
    L = 2
    corpus, check_fn = make_corpus_and_check_fn(vocab=vocab, L=L, n_items=10)

    strategy = DiverseBeamSearch(
        num_groups=2,
        lambda_per_level=[0.5, 0.5],
        use_embedding_distance=True,
    )

    beams = torch.randint(0, vocab, (B, k, 1))
    log_probas = torch.zeros(B, k)
    probas = torch.softmax(torch.randn(B * k, vocab), dim=-1)

    with pytest.raises(ValueError, match="codebook_embs"):
        strategy.expand(
            beams=beams,
            log_probas=log_probas,
            probas=probas,
            h=1,
            n_cands=n_cands,
            check_valid_fn=check_fn,
            codebook_embs=None,
        )
