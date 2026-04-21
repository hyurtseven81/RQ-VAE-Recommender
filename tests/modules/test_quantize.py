"""Regression tests for Quantize — esp. kmeans_initted persistence on reload.

The bug: kmeans_initted used to be a plain Python bool, so reloading a trained
checkpoint reset it to False and the next forward() re-ran KMeans on the first
batch, overwriting the trained codebook. Fixed by registering it as a buffer
and injecting True for legacy checkpoints via a load_state_dict pre-hook.
"""
import torch

from modules.quantize import Quantize, QuantizeForwardMode


def _fresh_quantize() -> Quantize:
    return Quantize(
        embed_dim=8,
        n_embed=16,
        forward_mode=QuantizeForwardMode.STE,
        do_kmeans_init=True,
    )


def test_kmeans_initted_is_buffer():
    q = _fresh_quantize()
    assert "kmeans_initted" in q.state_dict()
    assert "kmeans_initted" in dict(q.named_buffers())


def test_kmeans_initted_survives_round_trip():
    q_a = _fresh_quantize()
    q_a.kmeans_initted.fill_(True)
    q_b = _fresh_quantize()
    assert not q_b.kmeans_initted.item()
    q_b.load_state_dict(q_a.state_dict())
    assert q_b.kmeans_initted.item()


def test_legacy_checkpoint_treated_as_initted():
    """State dict without 'kmeans_initted' (old format) must load as initted=True."""
    q = _fresh_quantize()
    legacy_sd = {k: v for k, v in q.state_dict().items() if k != "kmeans_initted"}
    q_loaded = _fresh_quantize()
    q_loaded.load_state_dict(legacy_sd)
    assert q_loaded.kmeans_initted.item(), (
        "Pre-hook must inject True so the first forward pass does not reset "
        "the trained codebook via _kmeans_init."
    )


def test_forward_does_not_reinit_after_load():
    """After loading a trained codebook, the first forward() must not mutate it."""
    q = _fresh_quantize()
    # Simulate having been KMeans-initted during training
    q.kmeans_initted.fill_(True)
    trained_weight = torch.randn_like(q.embedding.weight)
    q.embedding.weight.data.copy_(trained_weight)

    x = torch.randn(4, 8)
    q.eval()
    _ = q(x, temperature=1.0)

    assert torch.allclose(q.embedding.weight.data, trained_weight), (
        "forward() must not overwrite the loaded codebook"
    )
