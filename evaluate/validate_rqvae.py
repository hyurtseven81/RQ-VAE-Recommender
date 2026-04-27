"""Validate an RQ-VAE checkpoint: unique SIDs, residual entropy, min codebook distance.

Runs on SageMaker. Inputs:
  --config_path   gin config (defines dataset + model hyperparams)
  --rqvae_checkpoint  path to the .pt file (or directory containing one)
  --output_dir    where to write verdict.json (default: SM_OUTPUT_DATA_DIR or /opt/ml/output/data)

Writes verdict.json with per-level stats and a boolean ``healthy`` flag.
A codebook is deemed collapsed when any level has unique_sids < 10 or min_dist < 0.01.
"""
import argparse
import json
import os
from pathlib import Path

import gin
import numpy as np
import torch
from torch.utils.data import BatchSampler, DataLoader, SequentialSampler

from data.processed import ItemData
from modules.rqvae import RqVae


def _find_checkpoint(path: str) -> Path:
    p = Path(path)
    if p.is_file():
        return p
    if p.is_dir():
        # Auto-extract any model.tar.gz (SageMaker doesn't unpack training-channel tarballs)
        tarballs = list(p.rglob("*.tar.gz"))
        if tarballs and not list(p.rglob("*.pt")):
            import tarfile
            extract_dir = Path("/tmp/ckpt_extracted")
            extract_dir.mkdir(parents=True, exist_ok=True)
            with tarfile.open(tarballs[0]) as tf:
                tf.extractall(extract_dir)
            p = extract_dir
        pts = sorted(p.rglob("*.pt"))
        if not pts:
            raise FileNotFoundError(f"No .pt file under {path}")
        # Prefer checkpoint_399999.pt, else latest
        for cand in pts:
            if cand.name == "checkpoint_399999.pt":
                return cand
        return pts[-1]
    raise FileNotFoundError(path)


def _read_train_params() -> dict:
    """Read gin-bound ``train_rqvae.train.*`` params needed to reconstruct RqVae
    and locate ItemData. Uses fully qualified selectors so this coexists with
    any other module that defines a configurable named ``train``.

    gin.query_parameter returns a ConfigurableReference for '%Enum.FOO' style
    values; we resolve those by calling .scoped_configurable_fn().
    """
    import gin.config
    import train_rqvae  # noqa: F401 — ensure @gin.configurable decorator registers

    def _q(name, default):
        try:
            v = gin.query_parameter(f"train_rqvae.train.{name}")
        except ValueError:
            return default
        if isinstance(v, gin.config.ConfigurableReference):
            return v.scoped_configurable_fn()
        return v

    return dict(
        input_dim=_q("vae_input_dim", 768),
        embed_dim=_q("vae_embed_dim", 32),
        hidden_dims=list(_q("vae_hidden_dims", (512, 256, 128))),
        codebook_size=_q("vae_codebook_size", 256),
        n_layers=_q("vae_n_layers", 3),
        n_cat_features=_q("vae_n_cat_feats", 0),
        codebook_normalize=_q("vae_codebook_normalize", False),
        codebook_sim_vq=_q("vae_sim_vq", False),
        codebook_mode=_q("vae_codebook_mode", None),
        codebook_kmeans_init=_q("use_kmeans_init", True),
        commitment_weight=_q("commitment_weight", 0.25),
        _dataset_folder=_q("dataset_folder", "dataset/amazon"),
        _dataset=_q("dataset", None),
        _dataset_split=_q("dataset_split", "beauty"),
    )


def validate(config_path: str, rqvae_checkpoint: str, output_dir: str,
             batch_size: int = 512, parse_gin: bool = True) -> dict:
    """Run validation and return the verdict dict. Also writes verdict.json to output_dir.

    If parse_gin is False, caller is expected to have already parsed the gin config
    (e.g. from a sanity-check entry that already called gin.parse_config_file()).
    """
    # Ensure train_rqvae's @gin.configurable registers before gin.parse_config_file
    # so the config's 'train.*' bindings resolve to train_rqvae.train.*.
    import train_rqvae  # noqa: F401
    if parse_gin:
        gin.parse_config_file(config_path)
    cfg = _read_train_params()
    dataset_folder = cfg.pop("_dataset_folder")
    dataset = cfg.pop("_dataset")
    dataset_split = cfg.pop("_dataset_split")

    # If SageMaker mounted a preprocessed dataset channel, link it into the
    # gin-configured dataset_folder so ItemData reads the cache instead of
    # re-downloading + re-embedding.
    sm_dataset_channel = os.environ.get("SM_CHANNEL_DATASET")
    if sm_dataset_channel:
        import shutil as _shutil
        target = Path(dataset_folder)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            if target.is_symlink() or target.is_file():
                target.unlink()
            else:
                _shutil.rmtree(target)
        target.symlink_to(sm_dataset_channel)
        print(f"Linked preprocessed dataset: {sm_dataset_channel} -> {target}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    ckpt_path = _find_checkpoint(rqvae_checkpoint)
    print(f"Loading checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_cfg = state.get("model_config", {})
    # Strip non-init keys from saved config (locals())
    model_cfg = {k: v for k, v in model_cfg.items() if k in RqVae.__init__.__code__.co_varnames}
    model_cfg.pop("self", None)
    model_cfg.pop("__class__", None)
    # Fall back to gin config if model_config is missing fields
    for k, v in cfg.items():
        model_cfg.setdefault(k, v)

    rqvae = RqVae(**model_cfg)
    rqvae.load_state_dict(state["model"])
    # Disable in-forward KMeans trigger so the trained codebook isn't overwritten
    # on the first call (kmeans_initted is a non-persisted Python attribute).
    for layer in rqvae.layers:
        layer.do_kmeans_init = False
        layer.kmeans_initted = True
    rqvae = rqvae.to(device).eval()
    n_layers = rqvae.n_layers
    codebook_size = rqvae.codebook_size
    print(f"Model: n_layers={n_layers}, codebook_size={codebook_size}")

    # Refuse to silently materialise raw data — ItemData would otherwise call
    # raw_data.process() when the cache is missing, which for ML32M downloads
    # 32M ratings + runs Sentence-T5 over ~86k items on CPU. Inside SageMaker
    # the SM_CHANNEL_DATASET symlink (set up above) populates this directory
    # before we get here; locally the operator must sync the preprocessed
    # cache from S3 first (see docs/runbook_operator.md §2).
    processed_dir = Path(dataset_folder) / "processed"
    processed_pts = (
        sorted(processed_dir.glob("*.pt")) if processed_dir.exists() else []
    )
    if not processed_pts:
        raise RuntimeError(
            f"processed cache missing under {processed_dir}/ — refusing to "
            f"trigger raw preprocessing. Sync with: "
            f"aws s3 sync $RQVAE_S3_BASE/datasets/{Path(dataset_folder).name}/ "
            f"{dataset_folder}/  (or run inside SageMaker where the "
            f"SM_CHANNEL_DATASET channel mounts the cache automatically)."
        )

    items = ItemData(
        root=dataset_folder,
        dataset=dataset,
        force_process=False,
        train_test_split="all",
        split=dataset_split,
    )
    print(f"Loaded {len(items)} items from dataset_split='{dataset_split}'")

    # Encode all items → collect SIDs + residual norms per level
    all_sids = [[] for _ in range(n_layers)]
    all_res_norms = [[] for _ in range(n_layers)]
    loader = DataLoader(
        items,
        sampler=BatchSampler(SequentialSampler(items), batch_size, False),
        batch_size=None,
        collate_fn=lambda b: b,
    )
    with torch.no_grad():
        for batch in loader:
            x = batch.x.to(device)
            out = rqvae.get_semantic_ids(x)
            # out.sem_ids: [L, B] or [B, L]; out.residuals: [L, D, B]
            sem_ids = out.sem_ids
            if sem_ids.ndim == 2 and sem_ids.shape[0] != n_layers:
                sem_ids = sem_ids.T  # normalize to [L, B]
            residuals = out.residuals  # [L, D, B]
            for lvl in range(n_layers):
                all_sids[lvl].append(sem_ids[lvl].cpu().numpy())
                all_res_norms[lvl].append(residuals[lvl].norm(dim=0).cpu().numpy())

    # Aggregate per-level stats
    per_level = []
    for lvl in range(n_layers):
        sids = np.concatenate(all_sids[lvl])
        norms = np.concatenate(all_res_norms[lvl])
        counts = np.bincount(sids, minlength=codebook_size)
        probs = counts[counts > 0] / counts.sum()
        entropy = float(-(probs * np.log2(probs)).sum())
        unique = int((counts > 0).sum())

        # Min pairwise distance on active codebook vectors
        layer = rqvae.layers[lvl]
        cb = layer.out_proj(layer.embedding.weight).detach()  # [K, D]
        active_mask = torch.zeros(codebook_size, dtype=torch.bool, device=cb.device)
        active_mask[torch.from_numpy(np.where(counts > 0)[0]).to(cb.device)] = True
        active_cb = cb[active_mask]
        if active_cb.shape[0] >= 2:
            dists = torch.cdist(active_cb, active_cb)
            dists.fill_diagonal_(float("inf"))
            min_dist = float(dists.min().item())
        else:
            min_dist = 0.0

        per_level.append(
            dict(
                level=lvl,
                unique_sids=unique,
                entropy_bits=entropy,
                mean_residual_norm=float(norms.mean()),
                std_residual_norm=float(norms.std()),
                min_codebook_distance=min_dist,
            )
        )
        print(
            f"L{lvl}: unique={unique}/{codebook_size}, entropy={entropy:.3f} bits, "
            f"min_dist={min_dist:.4f}, E|r|={norms.mean():.4f}"
        )

    collapsed_levels = [
        pl["level"]
        for pl in per_level
        if pl["unique_sids"] < 10 or pl["min_codebook_distance"] < 0.01
    ]
    healthy = len(collapsed_levels) == 0 and per_level[0]["entropy_bits"] > 2.0

    verdict = dict(
        healthy=healthy,
        checkpoint=str(ckpt_path),
        config=config_path,
        num_items=len(items),
        collapsed_levels=collapsed_levels,
        per_level=per_level,
    )
    os.makedirs(output_dir, exist_ok=True)
    out_file = Path(output_dir) / "verdict.json"
    with open(out_file, "w") as f:
        json.dump(verdict, f, indent=2)
    print(f"\n=== VERDICT: {'HEALTHY' if healthy else 'COLLAPSED'} ===")
    print(f"Wrote {out_file}")
    return verdict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--rqvae_checkpoint", required=True)
    parser.add_argument(
        "--output_dir",
        default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"),
    )
    parser.add_argument("--batch_size", type=int, default=512)
    args = parser.parse_args()
    validate(
        config_path=args.config_path,
        rqvae_checkpoint=args.rqvae_checkpoint,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
