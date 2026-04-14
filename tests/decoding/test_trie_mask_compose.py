"""Validity tests: all returned beams must be valid prefixes in the corpus.

Uses a synthetic 2-level, 4-code corpus where only 3 out of 16 full SIDs are valid.
Parametrized over VanillaBeamSearch, GumbelTopKBeamSearch, and HybridBeamSearch.
"""
import torch
import pytest

from modules.decoding.vanilla import VanillaBeamSearch
from modules.decoding.gumbel import GumbelTopKBeamSearch
from modules.decoding.hybrid import HybridBeamSearch


# ---------------------------------------------------------------------------
# Synthetic corpus: exactly 3 valid full SIDs out of 16 possible (4^2)
# ---------------------------------------------------------------------------

VALID_SIDS = torch.tensor([
    [0, 1],
    [1, 2],
    [3, 0],
])


def check_valid_prefix(prefix: torch.Tensor) -> torch.Tensor:
    depth = prefix.shape[1]
    trimmed = VALID_SIDS[:, :depth]
    return (trimmed.unsqueeze(1) == prefix.unsqueeze(0)).all(dim=2).any(dim=0)


# ---------------------------------------------------------------------------
# Strategy parametrize
# ---------------------------------------------------------------------------

STRATEGIES = [
    pytest.param(VanillaBeamSearch(), id="vanilla"),
    pytest.param(GumbelTopKBeamSearch(tau=1.0), id="gumbel_topk"),
    pytest.param(HybridBeamSearch(tau=1.0), id="hybrid"),
]


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_all_beams_valid_after_two_levels(strategy):
    """After two expand() calls, all beams should be valid full SIDs."""
    vocab = 4
    k = 3  # exactly 3 valid SIDs
    B = 2
    n_cands = 4  # all tokens

    torch.manual_seed(0)
    probas_h0 = torch.softmax(torch.randn(B, vocab), dim=-1)

    beams_h0, lp_h0, _ = strategy.expand(
        beams=None,
        log_probas=torch.zeros(B, k),
        probas=probas_h0,
        h=0,
        n_cands=n_cands,
        check_valid_fn=check_valid_prefix,
        codebook_embs=[torch.randn(vocab, 4), torch.randn(vocab, 4)],
    )

    assert beams_h0.shape == (B, k, 1), f"h=0 shape wrong: {beams_h0.shape}"

    # Verify h=0 beams are valid level-0 prefixes
    flat_h0 = beams_h0.reshape(-1, 1)
    valid_h0 = check_valid_prefix(flat_h0)
    # Some may be -inf (no valid), but at least the non-inf ones must be valid
    valid_mask_h0 = lp_h0.reshape(-1) != float("-inf")
    assert valid_h0[valid_mask_h0].all(), "Some h=0 beams are not valid prefixes"

    torch.manual_seed(1)
    probas_h1 = torch.softmax(torch.randn(B * k, vocab), dim=-1)

    beams_h1, lp_h1, _ = strategy.expand(
        beams=beams_h0,
        log_probas=lp_h0,
        probas=probas_h1,
        h=1,
        n_cands=n_cands,
        check_valid_fn=check_valid_prefix,
        codebook_embs=[torch.randn(vocab, 4), torch.randn(vocab, 4)],
    )

    assert beams_h1.shape == (B, k, 2), f"h=1 shape wrong: {beams_h1.shape}"

    # All non-inf beams must be valid full SIDs
    flat_h1 = beams_h1.reshape(-1, 2)
    valid_h1 = check_valid_prefix(flat_h1)
    valid_mask_h1 = lp_h1.reshape(-1) != float("-inf")

    assert valid_h1[valid_mask_h1].all(), (
        f"Strategy {type(strategy).__name__}: some final beams are not valid SIDs.\n"
        f"Beams: {flat_h1[valid_mask_h1]}\n"
        f"Valid: {valid_h1[valid_mask_h1]}"
    )
