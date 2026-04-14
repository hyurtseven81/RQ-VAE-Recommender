"""Tests for LevelAwareHybridDecoding.

Critical correctness guarantee: alpha=[0,...,0] must produce IDENTICAL beams
to the wrapped base strategy (VanillaBeamSearch, DBS, GumbelTopKBeamSearch).
"""
import torch
import torch.nn as nn
import pytest

from modules.decoding.vanilla import VanillaBeamSearch
from modules.decoding.dbs import DiverseBeamSearch
from modules.decoding.gumbel import GumbelTopKBeamSearch
from modules.decoding.level_aware_mix import LevelAwareHybridDecoding, AlphaParams


# ---------------------------------------------------------------------------
# Tiny synthetic AuxHead: identity-like MLP that maps d_model -> d_item
# ---------------------------------------------------------------------------

class TinyAuxHead(nn.Module):
    """Minimal aux head for testing: single linear layer, no dropout."""

    def __init__(self, d_model: int, d_item: int):
        super().__init__()
        self.proj = nn.Linear(d_model, d_item, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# ---------------------------------------------------------------------------
# Synthetic corpus: valid prefixes for trie mask tests
# ---------------------------------------------------------------------------

VALID_SIDS = torch.tensor([
    [0, 1],
    [1, 2],
    [3, 0],
    [2, 3],
    [0, 3],
])


def check_valid_prefix(prefix: torch.Tensor) -> torch.Tensor:
    depth = prefix.shape[1]
    trimmed = VALID_SIDS[:, :depth]
    return (trimmed.unsqueeze(1) == prefix.unsqueeze(0)).all(dim=2).any(dim=0)


def always_valid(prefix: torch.Tensor) -> torch.Tensor:
    return torch.ones(prefix.size(0), dtype=torch.bool)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_setup(vocab=8, d_model=16, d_item=8, B=2, k=3, L=2):
    """Return common test components."""
    torch.manual_seed(42)
    codebook_embs = [torch.randn(vocab, d_item) for _ in range(L)]
    aux_head = TinyAuxHead(d_model, d_item)
    # Disable randomness in aux_head for determinism
    nn.init.eye_(aux_head.proj.weight[:d_item, :d_item])
    return vocab, d_model, d_item, B, k, L, codebook_embs, aux_head


# ---------------------------------------------------------------------------
# Test: alpha=0 must reproduce base strategy output exactly
# ---------------------------------------------------------------------------

def test_alpha_zero_equiv_vanilla():
    """alpha=[0,0,...] must produce IDENTICAL beams to VanillaBeamSearch.

    This is the critical correctness test for z-score normalization:
    when alpha=0, LevelAwareHybridDecoding should pass the original probas
    through to the base strategy unchanged (bypassing the mixing path entirely).
    """
    vocab, d_model, d_item, B, k, L, codebook_embs, aux_head = make_setup()
    alpha_zeros = [0.0] * L

    vanilla = VanillaBeamSearch()
    level_aware = LevelAwareHybridDecoding(
        base_strategy=VanillaBeamSearch(),
        alpha=alpha_zeros,
        aux_head=aux_head,
    )

    # --- h=0 ---
    torch.manual_seed(7)
    probas_h0 = torch.softmax(torch.randn(B, vocab), dim=-1)
    decoder_hidden_h0 = torch.randn(B, d_model)
    log_probas_0 = torch.zeros(B, k)

    torch.manual_seed(99)
    beams_vanilla, lp_vanilla, _ = vanilla.expand(
        beams=None,
        log_probas=log_probas_0,
        probas=probas_h0.clone(),
        h=0,
        n_cands=vocab,
        check_valid_fn=always_valid,
        codebook_embs=codebook_embs,
    )

    torch.manual_seed(99)
    beams_mixed, lp_mixed, _ = level_aware.expand(
        beams=None,
        log_probas=log_probas_0,
        probas=probas_h0.clone(),
        h=0,
        n_cands=vocab,
        check_valid_fn=always_valid,
        codebook_embs=codebook_embs,
        decoder_hidden=decoder_hidden_h0,
    )

    assert torch.equal(beams_vanilla, beams_mixed), (
        f"alpha=0 h=0: beams differ.\nVanilla: {beams_vanilla}\nMixed: {beams_mixed}"
    )
    assert torch.allclose(lp_vanilla, lp_mixed), (
        f"alpha=0 h=0: log_probas differ.\nVanilla: {lp_vanilla}\nMixed: {lp_mixed}"
    )

    # --- h=1 ---
    torch.manual_seed(7)
    probas_h1 = torch.softmax(torch.randn(B * k, vocab), dim=-1)
    decoder_hidden_h1 = torch.randn(B * k, d_model)

    torch.manual_seed(99)
    beams2_vanilla, lp2_vanilla, _ = vanilla.expand(
        beams=beams_vanilla,
        log_probas=lp_vanilla,
        probas=probas_h1.clone(),
        h=1,
        n_cands=vocab,
        check_valid_fn=always_valid,
        codebook_embs=codebook_embs,
    )

    torch.manual_seed(99)
    beams2_mixed, lp2_mixed, _ = level_aware.expand(
        beams=beams_mixed,
        log_probas=lp_mixed,
        probas=probas_h1.clone(),
        h=1,
        n_cands=vocab,
        check_valid_fn=always_valid,
        codebook_embs=codebook_embs,
        decoder_hidden=decoder_hidden_h1,
    )

    assert torch.equal(beams2_vanilla, beams2_mixed), (
        f"alpha=0 h=1: beams differ.\nVanilla: {beams2_vanilla}\nMixed: {beams2_mixed}"
    )


def test_alpha_one_uses_dense():
    """alpha=[1,1,...] — beams should be ordered by dense score only.

    When alpha=1, the mixed probas are entirely determined by the dense score,
    so the top beam should correspond to the highest dot product between
    the query and partial reconstruction.
    """
    vocab, d_model, d_item, B, k, L, codebook_embs, aux_head = make_setup(
        vocab=8, d_model=8, d_item=8, B=1, k=4, L=2
    )
    alpha_ones = [1.0] * L

    level_aware = LevelAwareHybridDecoding(
        base_strategy=VanillaBeamSearch(),
        alpha=alpha_ones,
        aux_head=aux_head,
    )

    torch.manual_seed(0)
    probas_h0 = torch.softmax(torch.randn(B, vocab), dim=-1)
    decoder_hidden_h0 = torch.randn(B, d_model)

    beams, lp, _ = level_aware.expand(
        beams=None,
        log_probas=torch.zeros(B, k),
        probas=probas_h0,
        h=0,
        n_cands=vocab,
        check_valid_fn=always_valid,
        codebook_embs=codebook_embs,
        decoder_hidden=decoder_hidden_h0,
    )

    # Verify output shape
    assert beams.shape == (B, k, 1), f"Expected ({B}, {k}, 1), got {beams.shape}"

    # When alpha=1, mixed probas = softmax(z_score(dense_scores))
    # The top beam's token should have the highest dense score
    with torch.no_grad():
        q = aux_head(decoder_hidden_h0)  # [1, d_item]
    r = codebook_embs[0]  # [vocab, d_item]
    s_dense = (q @ r.T).squeeze(0)  # [vocab]

    top_token_by_dense = s_dense.argmax().item()
    top_beam_token = beams[0, 0, 0].item()
    # The top beam should match the top dense token
    assert top_beam_token == top_token_by_dense, (
        f"alpha=1: expected top token {top_token_by_dense}, got {top_beam_token}. "
        f"Dense scores: {s_dense}"
    )


def test_wraps_dbs():
    """LevelAwareHybridDecoding wrapping DBS runs without error."""
    vocab, d_model, d_item, B, k, L, codebook_embs, aux_head = make_setup(vocab=8, k=4)
    alpha = [0.3, 0.5]

    strategy = LevelAwareHybridDecoding(
        base_strategy=DiverseBeamSearch(num_groups=2),
        alpha=alpha,
        aux_head=aux_head,
    )

    torch.manual_seed(5)
    probas_h0 = torch.softmax(torch.randn(B, vocab), dim=-1)
    decoder_hidden_h0 = torch.randn(B, d_model)

    beams, lp, _ = strategy.expand(
        beams=None,
        log_probas=torch.zeros(B, k),
        probas=probas_h0,
        h=0,
        n_cands=vocab,
        check_valid_fn=always_valid,
        codebook_embs=codebook_embs,
        decoder_hidden=decoder_hidden_h0,
    )
    assert beams.shape == (B, k, 1)

    torch.manual_seed(6)
    probas_h1 = torch.softmax(torch.randn(B * k, vocab), dim=-1)
    decoder_hidden_h1 = torch.randn(B * k, d_model)

    beams2, lp2, _ = strategy.expand(
        beams=beams,
        log_probas=lp,
        probas=probas_h1,
        h=1,
        n_cands=vocab,
        check_valid_fn=always_valid,
        codebook_embs=codebook_embs,
        decoder_hidden=decoder_hidden_h1,
    )
    assert beams2.shape == (B, k, 2)


def test_wraps_gumbel():
    """LevelAwareHybridDecoding wrapping GumbelTopKBeamSearch runs without error."""
    vocab, d_model, d_item, B, k, L, codebook_embs, aux_head = make_setup()
    alpha = [0.4, 0.2]

    strategy = LevelAwareHybridDecoding(
        base_strategy=GumbelTopKBeamSearch(tau=0.5),
        alpha=alpha,
        aux_head=aux_head,
    )

    torch.manual_seed(10)
    probas_h0 = torch.softmax(torch.randn(B, vocab), dim=-1)
    decoder_hidden_h0 = torch.randn(B, d_model)

    beams, lp, _ = strategy.expand(
        beams=None,
        log_probas=torch.zeros(B, k),
        probas=probas_h0,
        h=0,
        n_cands=vocab,
        check_valid_fn=always_valid,
        codebook_embs=codebook_embs,
        decoder_hidden=decoder_hidden_h0,
    )
    assert beams.shape == (B, k, 1)

    torch.manual_seed(11)
    probas_h1 = torch.softmax(torch.randn(B * k, vocab), dim=-1)
    decoder_hidden_h1 = torch.randn(B * k, d_model)

    beams2, _, _ = strategy.expand(
        beams=beams,
        log_probas=lp,
        probas=probas_h1,
        h=1,
        n_cands=vocab,
        check_valid_fn=always_valid,
        codebook_embs=codebook_embs,
        decoder_hidden=decoder_hidden_h1,
    )
    assert beams2.shape == (B, k, 2)


def test_trie_mask_respected():
    """All returned beams are valid corpus prefixes even with mixing (alpha > 0)."""
    vocab = 4
    k = 3
    B = 2
    L = 2
    d_model = 8
    d_item = 4
    codebook_embs = [torch.randn(vocab, d_item) for _ in range(L)]
    aux_head = TinyAuxHead(d_model, d_item)
    alpha = [0.5, 0.5]

    strategy = LevelAwareHybridDecoding(
        base_strategy=VanillaBeamSearch(),
        alpha=alpha,
        aux_head=aux_head,
    )

    torch.manual_seed(20)
    probas_h0 = torch.softmax(torch.randn(B, vocab), dim=-1)
    decoder_hidden_h0 = torch.randn(B, d_model)

    beams, lp, _ = strategy.expand(
        beams=None,
        log_probas=torch.zeros(B, k),
        probas=probas_h0,
        h=0,
        n_cands=vocab,
        check_valid_fn=check_valid_prefix,
        codebook_embs=codebook_embs,
        decoder_hidden=decoder_hidden_h0,
    )

    # Non-inf beams must be valid h=0 prefixes
    flat = beams.reshape(-1, 1)
    valid = check_valid_prefix(flat)
    valid_mask = lp.reshape(-1) != float("-inf")
    assert valid[valid_mask].all(), (
        f"Some h=0 beams are invalid prefixes: {flat[valid_mask & ~valid]}"
    )

    torch.manual_seed(21)
    probas_h1 = torch.softmax(torch.randn(B * k, vocab), dim=-1)
    decoder_hidden_h1 = torch.randn(B * k, d_model)

    beams2, lp2, _ = strategy.expand(
        beams=beams,
        log_probas=lp,
        probas=probas_h1,
        h=1,
        n_cands=vocab,
        check_valid_fn=check_valid_prefix,
        codebook_embs=codebook_embs,
        decoder_hidden=decoder_hidden_h1,
    )

    flat2 = beams2.reshape(-1, 2)
    valid2 = check_valid_prefix(flat2)
    valid_mask2 = lp2.reshape(-1) != float("-inf")
    assert valid2[valid_mask2].all(), (
        f"Some h=1 beams are invalid: {flat2[valid_mask2 & ~valid2]}"
    )


def test_partial_reconstruction_shape():
    """r_h^cand has correct shape at h=0 and h>0."""
    vocab = 8
    d_item = 4
    B, k, L = 3, 5, 3
    d_model = 16

    codebook_embs = [torch.randn(vocab, d_item) for _ in range(L)]
    aux_head = TinyAuxHead(d_model, d_item)
    alpha = [0.3] * L

    strategy = LevelAwareHybridDecoding(
        base_strategy=VanillaBeamSearch(),
        alpha=alpha,
        aux_head=aux_head,
    )

    cand_ids = torch.arange(vocab)

    # h=0: beams is None → r should be [1, vocab, d_item]
    r_h0 = strategy._compute_partial_reconstruction(
        beams=None, cand_ids=cand_ids, codebook_embs=codebook_embs, h=0
    )
    assert r_h0.shape == (1, vocab, d_item), (
        f"h=0 reconstruction shape: expected (1, {vocab}, {d_item}), got {r_h0.shape}"
    )

    # h=1: beams is [B, k, 1] → r should be [B, k, vocab, d_item]
    beams_h1 = torch.randint(0, vocab, (B, k, 1))
    r_h1 = strategy._compute_partial_reconstruction(
        beams=beams_h1, cand_ids=cand_ids, codebook_embs=codebook_embs, h=1
    )
    assert r_h1.shape == (B, k, vocab, d_item), (
        f"h=1 reconstruction shape: expected ({B}, {k}, {vocab}, {d_item}), got {r_h1.shape}"
    )

    # h=2: beams is [B, k, 2] → r should be [B, k, vocab, d_item]
    beams_h2 = torch.randint(0, vocab, (B, k, 2))
    r_h2 = strategy._compute_partial_reconstruction(
        beams=beams_h2, cand_ids=cand_ids, codebook_embs=codebook_embs, h=2
    )
    assert r_h2.shape == (B, k, vocab, d_item), (
        f"h=2 reconstruction shape: expected ({B}, {k}, {vocab}, {d_item}), got {r_h2.shape}"
    )


def test_alpha_params_sigmoid():
    """AlphaParams: phi=0 -> alpha=0.5, phi large positive -> alpha~1."""
    params = AlphaParams(n_levels=3)
    # Default: phi=0 → all alphas = 0.5
    for a in params.alpha:
        assert abs(a - 0.5) < 1e-5, f"Expected 0.5, got {a}"

    # Set phi to large values
    with torch.no_grad():
        params.phi.fill_(100.0)
    for a in params.alpha:
        assert a > 0.99, f"Expected ~1.0, got {a}"

    # Set phi to large negative
    with torch.no_grad():
        params.phi.fill_(-100.0)
    for a in params.alpha:
        assert a < 0.01, f"Expected ~0.0, got {a}"
