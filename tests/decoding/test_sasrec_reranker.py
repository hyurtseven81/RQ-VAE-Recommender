"""Tests for SASRecReranker post-hoc reranking.

Key invariants:
- alpha=0: reranked order matches original log_proba order (dense ignored)
- alpha=1: reranked order matches dense score order (AR score ignored)
"""
import torch
import torch.nn as nn
import pytest

from modules.decoding.sasrec_reranker import SASRecReranker


# ---------------------------------------------------------------------------
# Tiny AuxHead for testing
# ---------------------------------------------------------------------------

class TinyAuxHead(nn.Module):
    def __init__(self, d_model: int, d_item: int):
        super().__init__()
        self.proj = nn.Linear(d_model, d_item, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_setup(B=2, k=5, L=3, vocab=8, d_model=16, d_item=8):
    torch.manual_seed(42)
    codebook_embs = [torch.randn(vocab, d_item) for _ in range(L)]
    aux_head = TinyAuxHead(d_model, d_item)
    beams = torch.randint(0, vocab, (B, k, L))
    log_probas = torch.randn(B, k)
    decoder_hidden = torch.randn(B, d_model)
    return codebook_embs, aux_head, beams, log_probas, decoder_hidden


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_output_shapes():
    """Reranker returns (beams, scores) with correct shapes."""
    codebook_embs, aux_head, beams, log_probas, decoder_hidden = make_setup()
    B, k, L = beams.shape

    reranker = SASRecReranker(aux_head=aux_head, alpha=0.5)
    reranked_beams, reranked_scores = reranker.rerank(
        beams=beams,
        log_probas=log_probas,
        codebook_embs=codebook_embs,
        decoder_hidden=decoder_hidden,
    )

    assert reranked_beams.shape == (B, k, L), (
        f"Expected ({B}, {k}, {L}), got {reranked_beams.shape}"
    )
    assert reranked_scores.shape == (B, k), (
        f"Expected ({B}, {k}), got {reranked_scores.shape}"
    )


def test_alpha_zero_preserves_order():
    """alpha=0 → reranked order matches original log_proba order.

    When alpha=0, mixed = z_score(log_probas), which is a monotone transform
    of log_probas, so the sort order must match the original log_proba ranking.
    """
    codebook_embs, aux_head, beams, log_probas, decoder_hidden = make_setup()
    B, k, L = beams.shape

    reranker = SASRecReranker(aux_head=aux_head, alpha=0.0)
    reranked_beams, _ = reranker.rerank(
        beams=beams,
        log_probas=log_probas,
        codebook_embs=codebook_embs,
        decoder_hidden=decoder_hidden,
    )

    # Sort original beams by log_probas descending
    _, expected_order = log_probas.sort(dim=-1, descending=True)
    expected_beams = torch.gather(
        beams, 1, expected_order.unsqueeze(-1).expand(-1, -1, L)
    )

    assert torch.equal(reranked_beams, expected_beams), (
        f"alpha=0: reranked order does not match log_proba order.\n"
        f"Expected order: {expected_order}\n"
        f"Reranked beams: {reranked_beams}\n"
        f"Expected beams: {expected_beams}"
    )


def test_alpha_one_uses_dense_only():
    """alpha=1 → reranked order matches dense score order.

    When alpha=1, mixed = z_score(s_dense), which is a monotone transform
    of s_dense, so the sort order must match the dense score ranking.
    """
    codebook_embs, aux_head, beams, log_probas, decoder_hidden = make_setup()
    B, k, L = beams.shape

    reranker = SASRecReranker(aux_head=aux_head, alpha=1.0)
    reranked_beams, _ = reranker.rerank(
        beams=beams,
        log_probas=log_probas,
        codebook_embs=codebook_embs,
        decoder_hidden=decoder_hidden,
    )

    # Compute expected dense score order
    with torch.no_grad():
        q = aux_head(decoder_hidden)  # [B, d_item]
    r_full = sum(
        codebook_embs[l][beams[:, :, l]] for l in range(L)
    )  # [B, k, d_item]
    s_dense = (q.unsqueeze(1) * r_full).sum(-1)  # [B, k]

    _, expected_order = s_dense.sort(dim=-1, descending=True)
    expected_beams = torch.gather(
        beams, 1, expected_order.unsqueeze(-1).expand(-1, -1, L)
    )

    assert torch.equal(reranked_beams, expected_beams), (
        f"alpha=1: reranked order does not match dense score order.\n"
        f"Expected order: {expected_order}\n"
        f"Reranked beams: {reranked_beams}\n"
        f"Expected beams: {expected_beams}"
    )


def test_all_beams_present():
    """Reranking is a permutation: all original beams must appear in output."""
    codebook_embs, aux_head, beams, log_probas, decoder_hidden = make_setup()
    B, k, L = beams.shape

    reranker = SASRecReranker(aux_head=aux_head, alpha=0.5)
    reranked_beams, _ = reranker.rerank(
        beams=beams,
        log_probas=log_probas,
        codebook_embs=codebook_embs,
        decoder_hidden=decoder_hidden,
    )

    # For each batch, check that every original beam appears in reranked output
    for b in range(B):
        original = set(tuple(beams[b, i].tolist()) for i in range(k))
        reranked = set(tuple(reranked_beams[b, i].tolist()) for i in range(k))
        assert original == reranked, (
            f"Batch {b}: reranking lost beams.\nOriginal: {original}\nReranked: {reranked}"
        )


def test_scores_descending():
    """Returned scores should be in descending order (best beam first)."""
    codebook_embs, aux_head, beams, log_probas, decoder_hidden = make_setup()

    for alpha in [0.0, 0.3, 0.5, 0.7, 1.0]:
        reranker = SASRecReranker(aux_head=aux_head, alpha=alpha)
        _, scores = reranker.rerank(
            beams=beams,
            log_probas=log_probas,
            codebook_embs=codebook_embs,
            decoder_hidden=decoder_hidden,
        )
        # Check that scores[b, 0] >= scores[b, 1] >= ... for each b
        assert (scores[:, :-1] >= scores[:, 1:]).all(), (
            f"alpha={alpha}: scores not in descending order: {scores}"
        )


def test_single_beam():
    """Reranker works correctly with k=1 (trivial permutation)."""
    B, k, L = 3, 1, 2
    vocab, d_model, d_item = 4, 8, 4
    torch.manual_seed(0)
    codebook_embs = [torch.randn(vocab, d_item) for _ in range(L)]
    aux_head = TinyAuxHead(d_model, d_item)
    beams = torch.randint(0, vocab, (B, k, L))
    log_probas = torch.randn(B, k)
    decoder_hidden = torch.randn(B, d_model)

    reranker = SASRecReranker(aux_head=aux_head, alpha=0.5)
    reranked_beams, scores = reranker.rerank(
        beams=beams,
        log_probas=log_probas,
        codebook_embs=codebook_embs,
        decoder_hidden=decoder_hidden,
    )
    assert reranked_beams.shape == (B, k, L)
    assert scores.shape == (B, k)
    assert torch.equal(reranked_beams, beams), "k=1: beam should be unchanged"
