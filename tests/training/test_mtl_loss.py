"""Tests for MTL loss functions and LambdaWarmupSchedule."""

import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from modules.heads.mtl_losses import LambdaWarmupSchedule, sasrec_infonce_loss
from modules.heads.sasrec_head import SASRecAuxHead


# ------------------------------------------------------------------ #
# sasrec_infonce_loss tests                                            #
# ------------------------------------------------------------------ #

def test_infonce_loss_shape():
    """Loss should be a scalar tensor."""
    B, d_item, num_items = 8, 16, 200
    query = torch.randn(B, d_item)
    item_emb = torch.nn.Embedding(num_items, d_item)
    pos_ids = torch.randint(0, num_items, (B,))

    loss = sasrec_infonce_loss(
        query=query,
        positive_item_ids=pos_ids,
        item_embeddings=item_emb,
        n_negatives=32,
        temperature=0.05,
    )

    assert loss.shape == (), f"Expected scalar, got shape {loss.shape}"
    assert loss.item() > 0.0, "Loss should be positive"


def test_infonce_loss_positive_case():
    """When positive score >> all negatives, loss should be near 0."""
    B, d_item, num_items = 4, 16, 10000  # large corpus avoids collision with positives

    # Each query/positive pair uses a unique dimension so in-batch cross-scores are 0.
    query = torch.zeros(B, d_item)
    item_emb = torch.nn.Embedding(num_items, d_item)
    torch.nn.init.zeros_(item_emb.weight)
    pos_ids = torch.arange(B)
    with torch.no_grad():
        for i in range(B):
            query[i, i] = 100.0          # large component on unique dim
            item_emb.weight[i, i] = 1.0  # positive: perfectly aligned
        item_emb.weight[B:] = -0.1       # random negatives: anti-aligned

    loss = sasrec_infonce_loss(
        query=query,
        positive_item_ids=pos_ids,
        item_embeddings=item_emb,
        n_negatives=16,
        temperature=0.05,
    )

    assert loss.item() < 0.1, (
        f"Expected near-zero loss when positives dominate, got {loss.item():.4f}"
    )


def test_infonce_loss_differentiable():
    """Loss should support backward pass (gradients flow to query)."""
    B, d_item, num_items = 4, 8, 50
    query = torch.randn(B, d_item, requires_grad=True)
    item_emb = torch.nn.Embedding(num_items, d_item)
    pos_ids = torch.randint(0, num_items, (B,))

    loss = sasrec_infonce_loss(
        query=query,
        positive_item_ids=pos_ids,
        item_embeddings=item_emb,
        n_negatives=16,
        temperature=0.05,
    )
    loss.backward()

    assert query.grad is not None, "Gradient did not flow back to query"


# ------------------------------------------------------------------ #
# LambdaWarmupSchedule tests                                           #
# ------------------------------------------------------------------ #

def test_lambda_warmup_zero_at_start():
    """get(0) should return 0.0 when warmup_steps > 0."""
    schedule = LambdaWarmupSchedule(lambda_max=0.2, warmup_steps=1000)
    assert schedule.get(0) == 0.0, f"Expected 0.0 at step 0, got {schedule.get(0)}"


def test_lambda_warmup_max_at_end():
    """get(warmup_steps) should return lambda_max."""
    schedule = LambdaWarmupSchedule(lambda_max=0.2, warmup_steps=1000)
    result = schedule.get(1000)
    assert result == pytest.approx(0.2), f"Expected 0.2 at step 1000, got {result}"


def test_lambda_warmup_linear():
    """get(500) should return lambda_max / 2 for warmup_steps=1000."""
    schedule = LambdaWarmupSchedule(lambda_max=0.2, warmup_steps=1000)
    result = schedule.get(500)
    assert result == pytest.approx(0.1), f"Expected 0.1 at step 500, got {result}"


def test_lambda_warmup_past_end():
    """get(step > warmup_steps) should be capped at lambda_max."""
    schedule = LambdaWarmupSchedule(lambda_max=0.2, warmup_steps=1000)
    result = schedule.get(2000)
    assert result == pytest.approx(0.2), f"Expected 0.2 for step > warmup_steps, got {result}"


def test_lambda_warmup_zero_warmup_steps():
    """When warmup_steps=0, get(any step) should return lambda_max immediately."""
    schedule = LambdaWarmupSchedule(lambda_max=0.5, warmup_steps=0)
    assert schedule.get(0) == pytest.approx(0.5)
    assert schedule.get(100) == pytest.approx(0.5)


# ------------------------------------------------------------------ #
# Per-position aux supervision (audit bug B1 fix)                     #
# ------------------------------------------------------------------ #

def test_per_position_aux_supervision_yields_gradients_at_every_level():
    """MTL must propagate gradient from every decoder position, not just
    the final one.

    Mirrors the per-level loop in ``train_decoder_mtl.py``: at each h we
    query the aux head on ``decoder_output_full[:, h, :]`` and sum the
    InfoNCE losses. The resulting gradient on decoder_output_full must be
    non-zero at every slice h ∈ [0, L), which was not the case when only
    the final position was supervised (the pre-fix behaviour).
    """
    torch.manual_seed(0)
    B, L, d_model, d_item, num_items = 4, 3, 16, 8, 64

    decoder_output_full = torch.randn(B, L, d_model, requires_grad=True)
    aux_head = SASRecAuxHead(d_model=d_model, d_item=d_item, num_items=num_items)
    pos_ids = torch.randint(0, num_items, (B,))

    aux_loss = torch.tensor(0.0)
    for h in range(L):
        query_h = aux_head(decoder_output_full[:, h, :])
        aux_loss = aux_loss + sasrec_infonce_loss(
            query=query_h,
            positive_item_ids=pos_ids,
            item_embeddings=aux_head.item_embeddings,
            n_negatives=16,
            temperature=0.05,
        )
    aux_loss = aux_loss / L
    aux_loss.backward()

    # Gradient must be non-zero at every slice h ∈ [0, L).
    grad = decoder_output_full.grad
    assert grad is not None, "decoder_output_full.grad is None"
    for h in range(L):
        slice_norm = grad[:, h, :].norm().item()
        assert slice_norm > 0.0, (
            f"B1 regression: no gradient at decoder position h={h}. "
            f"Per-position aux supervision is not wired up."
        )
