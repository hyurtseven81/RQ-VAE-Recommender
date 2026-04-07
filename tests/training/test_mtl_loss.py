"""Tests for MTL loss functions and LambdaWarmupSchedule."""

import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from modules.heads.mtl_losses import sasrec_infonce_loss, LambdaWarmupSchedule


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
    B, d_item, num_items = 4, 16, 100

    # Construct query and embeddings so that positive dot-product is huge
    query = torch.zeros(B, d_item)
    query[:, 0] = 100.0  # large component in dim 0

    item_emb = torch.nn.Embedding(num_items, d_item)
    torch.nn.init.zeros_(item_emb.weight)
    # Make positive item embeddings aligned with query
    pos_ids = torch.arange(B)
    with torch.no_grad():
        item_emb.weight[pos_ids, 0] = 1.0   # aligned
        item_emb.weight[B:, 0] = -1.0       # negatives are anti-aligned

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
