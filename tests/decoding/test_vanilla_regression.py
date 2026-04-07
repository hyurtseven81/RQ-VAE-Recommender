"""Regression test: VanillaBeamSearch must produce identical results to the
original inline generate() logic.

The test constructs a minimal synthetic setup (no real model needed) that
exercises exactly the logic paths in VanillaBeamSearch.expand() against a
reference implementation of the same algorithm, given identical random state.
"""
import torch
import pytest

from modules.decoding.vanilla import VanillaBeamSearch


# ---------------------------------------------------------------------------
# Reference implementation — copied verbatim from original model.py generate()
# ---------------------------------------------------------------------------

def _reference_expand_h0(probas, n_cands, k, check_valid_fn):
    """Original h==0 path."""
    B = probas.size(0)
    samples = torch.multinomial(probas, num_samples=n_cands)
    samp_log_p = torch.log(torch.gather(probas, 1, samples))
    is_valid = check_valid_fn(samples.reshape(-1, 1)).reshape(B, n_cands)
    scores, idx = samp_log_p.masked_fill(~is_valid, float("-inf")).sort(-1, descending=True)
    top_k_idx = idx[:, :k]
    generated = torch.gather(samples, 1, top_k_idx).unsqueeze(-1)  # [B, k, 1]
    log_probas = scores[:, :k]
    return generated, log_probas


def _reference_expand_hN(probas, generated, log_probas, n_cands, k, check_valid_fn):
    """Original h>0 path."""
    B, _, h = generated.shape
    samples = torch.multinomial(probas, num_samples=n_cands)
    samp_log_p = torch.log(torch.gather(probas, 1, samples))
    prev = generated.reshape(-1, h).repeat_interleave(n_cands, dim=0)
    prefix = torch.cat([prev, samples.reshape(-1, 1)], dim=1)
    is_valid = check_valid_fn(prefix).reshape(B, k * n_cands)
    scores, idx = (
        (
            samp_log_p.reshape(B, k * n_cands)
            + log_probas.repeat_interleave(n_cands, dim=1)
        )
        .masked_fill(~is_valid, float("-inf"))
        .sort(-1, descending=True)
    )
    top_k_idx = idx[:, :k]
    parent_beam_idx = top_k_idx // n_cands
    parent_global = (
        parent_beam_idx
        + torch.arange(B, device=parent_beam_idx.device).unsqueeze(1) * k
    ).flatten()
    parent_ids = torch.gather(
        generated, 1, parent_beam_idx.unsqueeze(-1).expand(-1, -1, h)
    )
    new_ids = torch.gather(
        samples.reshape(B, k * n_cands), 1, top_k_idx
    ).unsqueeze(-1)
    new_generated = torch.cat([parent_ids, new_ids], dim=-1)
    new_log_probas = scores[:, :k]
    return new_generated, new_log_probas, parent_global


# ---------------------------------------------------------------------------
# Synthetic corpus & check_valid_fn
# ---------------------------------------------------------------------------

def make_corpus_and_check_fn(vocab: int, L: int, n_items: int, seed: int = 42):
    """Create a random corpus of SIDs and a corresponding check_valid_fn."""
    rng = torch.Generator()
    rng.manual_seed(seed)
    corpus = torch.randint(0, vocab, (n_items, L), generator=rng)
    # Deduplicate
    corpus = torch.unique(corpus, dim=0)

    def check_valid_prefix(prefix: torch.Tensor) -> torch.Tensor:
        depth = prefix.shape[1]
        trimmed = corpus[:, :depth]
        return (trimmed.unsqueeze(1) == prefix.unsqueeze(0)).all(dim=2).any(dim=0)

    return corpus, check_valid_prefix


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("L", [2, 3])
@pytest.mark.parametrize("B", [1, 3])
def test_vanilla_parity_h0(L, B):
    """h=0 path: VanillaBeamSearch.expand matches reference."""
    vocab = 8
    k = 4
    n_cands = 6
    _, check_fn = make_corpus_and_check_fn(vocab=vocab, L=L, n_items=20)

    strategy = VanillaBeamSearch()

    # Generate identical probas
    torch.manual_seed(0)
    probas = torch.softmax(torch.randn(B, vocab), dim=-1)
    log_probas_init = torch.zeros(B, k)

    # Reference
    torch.manual_seed(1)
    ref_generated, ref_log_probas = _reference_expand_h0(
        probas.clone(), n_cands=n_cands, k=k, check_valid_fn=check_fn
    )

    # VanillaBeamSearch
    torch.manual_seed(1)
    new_beams, new_log_probas, parent_idx = strategy.expand(
        beams=None,
        log_probas=log_probas_init,
        probas=probas.clone(),
        h=0,
        n_cands=n_cands,
        check_valid_fn=check_fn,
        codebook_embs=[],
    )

    assert torch.allclose(new_beams.float(), ref_generated.float()), \
        f"Beams mismatch at h=0:\n  ref={ref_generated}\n  got={new_beams}"
    assert torch.allclose(new_log_probas, ref_log_probas), \
        f"Log-probas mismatch at h=0:\n  ref={ref_log_probas}\n  got={new_log_probas}"
    assert parent_idx.numel() == 0  # no reorder at h=0


@pytest.mark.parametrize("B", [1, 2])
def test_vanilla_parity_h1(B):
    """h=1 path: VanillaBeamSearch.expand matches reference."""
    vocab = 8
    k = 4
    L = 2
    n_cands = 6
    _, check_fn = make_corpus_and_check_fn(vocab=vocab, L=L, n_items=30)

    # Build synthetic h=0 beams
    torch.manual_seed(2)
    beams = torch.randint(0, vocab, (B, k, 1))
    log_probas = torch.randn(B, k)

    # probas for h=1
    torch.manual_seed(3)
    probas = torch.softmax(torch.randn(B * k, vocab), dim=-1)

    strategy = VanillaBeamSearch()

    # Reference
    torch.manual_seed(4)
    ref_generated, ref_log_probas, ref_parent = _reference_expand_hN(
        probas.clone(), beams.clone(), log_probas.clone(),
        n_cands=n_cands, k=k, check_valid_fn=check_fn
    )

    # VanillaBeamSearch
    torch.manual_seed(4)
    new_beams, new_log_probas, parent_idx = strategy.expand(
        beams=beams.clone(),
        log_probas=log_probas.clone(),
        probas=probas.clone(),
        h=1,
        n_cands=n_cands,
        check_valid_fn=check_fn,
        codebook_embs=[],
    )

    assert torch.allclose(new_beams.float(), ref_generated.float()), \
        f"Beams mismatch at h=1:\n  ref={ref_generated}\n  got={new_beams}"
    assert torch.allclose(new_log_probas, ref_log_probas), \
        f"Log-probas mismatch at h=1:\n  ref={ref_log_probas}\n  got={new_log_probas}"
    assert torch.equal(parent_idx, ref_parent), \
        f"Parent idx mismatch:\n  ref={ref_parent}\n  got={parent_idx}"


def test_vanilla_returns_k_beams():
    """Output beam count equals k."""
    vocab, k, B, n_cands = 8, 3, 2, 5
    _, check_fn = make_corpus_and_check_fn(vocab=vocab, L=2, n_items=20)
    probas = torch.softmax(torch.randn(B, vocab), dim=-1)
    strategy = VanillaBeamSearch()

    new_beams, new_log_probas, _ = strategy.expand(
        beams=None,
        log_probas=torch.zeros(B, k),
        probas=probas,
        h=0,
        n_cands=n_cands,
        check_valid_fn=check_fn,
        codebook_embs=[],
    )
    assert new_beams.shape == (B, k, 1)
    assert new_log_probas.shape == (B, k)
