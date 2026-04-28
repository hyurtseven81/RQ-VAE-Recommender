"""Resolve decoder + RQ-VAE architecture kwargs from a parsed gin config.

Eval entries (`run_eval`, `alpha_search`, `alpha_train`) need to know the
architecture used at training time so they reconstruct the model with
matching shapes. Without this helper, those entries used hard-coded
function defaults (Amazon Beauty values) that silently mismatched
non-Amazon datasets — e.g. ML32M's ``vae_embed_dim=64`` would be
ignored, the decoder would be built at ``embed_dim=32``, and
``load_state_dict`` would surface the mismatch as a shape error.

Supports both the vanilla (`train.*`) and MTL (`train_mtl.*`) gin
binding scopes; tries each in turn and falls back to the Amazon defaults
if neither is bound.
"""
from __future__ import annotations

import gin

# Amazon-Beauty canonical defaults (same as the original `_get_arch` in
# run_eval.py / alpha_search.py / alpha_train.py).
_DEFAULTS = dict(
    vae_input_dim=768,
    vae_hidden_dims=[512, 256, 128],
    vae_embed_dim=32,
    vae_n_cat_feats=0,
    vae_codebook_size=256,
    vae_n_layers=3,
    vae_codebook_normalize=False,
    vae_sim_vq=False,
    t5_d_model=384,
    t5_num_heads=6,
    t5_d_ff=1024,
    t5_num_layers=4,
    top_k_for_generation=10,
    should_add_sep_token=True,
)


def _query(name: str, default):
    """Look up a gin-bound parameter under either MTL or vanilla scope.

    MTL gin configs bind ``train_mtl.<name>``; vanilla ones bind
    ``train.<name>``. Either short form is acceptable — gin canonicalises.
    """
    for scope in ("train_mtl", "train"):
        try:
            v = gin.query_parameter(f"{scope}.{name}")
        except (ValueError, KeyError):
            continue
        # Resolve enum-style references (rare for arch kwargs, harmless to keep).
        if isinstance(v, gin.config.ConfigurableReference):
            return v.scoped_configurable_fn()
        return v
    return default


def get_arch() -> dict:
    """Return the architecture kwargs needed to reconstruct the decoder."""
    out = {k: _query(k, default) for k, default in _DEFAULTS.items()}
    # gin returns tuples for list-typed params; coerce so downstream
    # MLP construction (which expects a list) doesn't trip on .index() etc.
    if isinstance(out["vae_hidden_dims"], tuple):
        out["vae_hidden_dims"] = list(out["vae_hidden_dims"])
    return out
