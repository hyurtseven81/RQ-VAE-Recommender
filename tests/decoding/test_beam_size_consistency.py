"""Beam size consistency: returned beam count == k for all strategies.

Parametrized over all four strategies (when enough valid paths exist in the trie).
"""
import torch
import pytest

from modules.decoding.vanilla import VanillaBeamSearch
from modules.decoding.dbs import DiverseBeamSearch
from modules.decoding.gumbel import GumbelTopKBeamSearch
from modules.decoding.hybrid import HybridBeamSearch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_full_corpus(vocab: int, L: int):
    """All vocab^L SIDs are valid — guarantees k valid paths always exist."""
    import itertools
    all_sids = list(itertools.product(range(vocab), repeat=L))
    return torch.tensor(all_sids, dtype=torch.long)


def make_check_fn(corpus):
    def check_valid_prefix(prefix: torch.Tensor) -> torch.Tensor:
        depth = prefix.shape[1]
        trimmed = corpus[:, :depth]
        return (trimmed.unsqueeze(1) == prefix.unsqueeze(0)).all(dim=2).any(dim=0)
    return check_valid_prefix


def make_codebook_embs(L: int, vocab: int, embed_dim: int = 4):
    return [torch.randn(vocab, embed_dim) for _ in range(L)]


# ---------------------------------------------------------------------------
# Strategy fixtures
# ---------------------------------------------------------------------------

STRATEGIES = [
    pytest.param(lambda k: VanillaBeamSearch(), id="vanilla"),
    pytest.param(lambda k: DiverseBeamSearch(num_groups=2, lambda_per_level=[0.5, 0.5]), id="dbs"),
    pytest.param(lambda k: GumbelTopKBeamSearch(tau=1.0), id="gumbel_topk"),
    pytest.param(lambda k: HybridBeamSearch(k_det=None, tau=1.0), id="hybrid"),
]


@pytest.mark.parametrize("strategy_factory", STRATEGIES)
@pytest.mark.parametrize("k", [2, 4, 6])
def test_beam_count_h0(strategy_factory, k):
    """Beam count at h=0 == k."""
    vocab = 8
    L = 2
    B = 2
    n_cands = min(8, vocab)
    corpus = make_full_corpus(vocab=vocab, L=L)
    check_fn = make_check_fn(corpus)
    embs = make_codebook_embs(L=L, vocab=vocab)
    strategy = strategy_factory(k)

    torch.manual_seed(0)
    probas = torch.softmax(torch.randn(B, vocab), dim=-1)
    beams, lp, _ = strategy.expand(
        beams=None,
        log_probas=torch.zeros(B, k),
        probas=probas,
        h=0,
        n_cands=n_cands,
        check_valid_fn=check_fn,
        codebook_embs=embs,
    )
    assert beams.shape[1] == k, f"Expected k={k} beams, got {beams.shape[1]}"
    assert beams.shape == (B, k, 1)
    assert lp.shape == (B, k)


@pytest.mark.parametrize("strategy_factory", STRATEGIES)
@pytest.mark.parametrize("k", [2, 4, 6])
def test_beam_count_h1(strategy_factory, k):
    """Beam count at h=1 == k."""
    vocab = 8
    L = 2
    B = 2
    n_cands = min(8, vocab)
    corpus = make_full_corpus(vocab=vocab, L=L)
    check_fn = make_check_fn(corpus)
    embs = make_codebook_embs(L=L, vocab=vocab)
    strategy = strategy_factory(k)

    # First get valid h=0 beams
    torch.manual_seed(0)
    probas_h0 = torch.softmax(torch.randn(B, vocab), dim=-1)
    beams_h0, lp_h0, _ = strategy.expand(
        beams=None,
        log_probas=torch.zeros(B, k),
        probas=probas_h0,
        h=0,
        n_cands=n_cands,
        check_valid_fn=check_fn,
        codebook_embs=embs,
    )

    torch.manual_seed(1)
    probas_h1 = torch.softmax(torch.randn(B * k, vocab), dim=-1)
    beams_h1, lp_h1, parent_idx = strategy.expand(
        beams=beams_h0,
        log_probas=lp_h0,
        probas=probas_h1,
        h=1,
        n_cands=n_cands,
        check_valid_fn=check_fn,
        codebook_embs=embs,
    )
    assert beams_h1.shape[1] == k, f"Expected k={k} beams, got {beams_h1.shape[1]}"
    assert beams_h1.shape == (B, k, 2)
    assert lp_h1.shape == (B, k)
    assert parent_idx.shape == (B * k,)
