"""Tests for SASRecAuxHead and build_item_embeddings_from_rqvae."""

import pytest
import torch
import sys
import os

# Ensure repo root is on sys.path when running pytest from any directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from modules.heads.sasrec_head import SASRecAuxHead, build_item_embeddings_from_rqvae


# ------------------------------------------------------------------ #
# Fixtures                                                             #
# ------------------------------------------------------------------ #

@pytest.fixture
def small_head():
    return SASRecAuxHead(d_model=64, d_item=16, num_items=100)


# ------------------------------------------------------------------ #
# Tests                                                                #
# ------------------------------------------------------------------ #

def test_forward_shape(small_head):
    """forward([B, d_model]) -> [B, d_item]."""
    x = torch.randn(4, 64)
    out = small_head(x)
    assert out.shape == (4, 16), f"Expected (4, 16), got {out.shape}"


def test_score_all_items_shape(small_head):
    """score_all_items([B, d_item]) -> [B, num_items]."""
    query = torch.randn(4, 16)
    scores = small_head.score_all_items(query)
    assert scores.shape == (4, 100), f"Expected (4, 100), got {scores.shape}"


def test_score_items_shape(small_head):
    """score_items([B, d_item], [B, N]) -> [B, N]."""
    query = torch.randn(4, 16)
    item_ids = torch.randint(0, 100, (4, 5))
    scores = small_head.score_items(query, item_ids)
    assert scores.shape == (4, 5), f"Expected (4, 5), got {scores.shape}"


def test_embedding_init_from_tensor():
    """item_embeddings.weight should match the provided init tensor."""
    init = torch.randn(50, 16)
    head = SASRecAuxHead(d_model=64, d_item=16, num_items=50, item_embeddings_init=init)
    assert torch.allclose(head.item_embeddings.weight, init), (
        "item_embeddings.weight does not match provided init tensor"
    )


def test_embedding_init_none_does_not_crash():
    """Creating without init tensor should not raise."""
    head = SASRecAuxHead(d_model=64, d_item=16, num_items=50, item_embeddings_init=None)
    assert head.item_embeddings.weight.shape == (50, 16)


def test_gradient_flows(small_head):
    """Backward pass should populate gradients on proj1 and proj2."""
    x = torch.randn(4, 64)
    query = small_head(x)
    loss = query.sum()
    loss.backward()

    assert small_head.proj1.weight.grad is not None, "proj1.weight has no gradient"
    assert small_head.proj2.weight.grad is not None, "proj2.weight has no gradient"


def test_build_from_rqvae():
    """build_item_embeddings_from_rqvae should return [num_items, embed_dim]."""
    pytest.importorskip("einops", reason="einops not installed — skipping RqVae integration test")
    from modules.rqvae import RqVae

    embed_dim = 8
    input_dim = 16
    rqvae = RqVae(
        input_dim=input_dim,
        embed_dim=embed_dim,
        hidden_dims=[12],
        codebook_size=4,
        codebook_kmeans_init=False,
        n_layers=2,
        n_cat_features=0,
    )

    num_items = 20
    item_features = torch.randn(num_items, input_dim)

    embeddings = build_item_embeddings_from_rqvae(
        rqvae=rqvae,
        item_features=item_features,
        batch_size=8,
        device="cpu",
    )

    assert embeddings.shape == (num_items, embed_dim), (
        f"Expected ({num_items}, {embed_dim}), got {embeddings.shape}"
    )
    # Should be on CPU
    assert embeddings.device.type == "cpu"
