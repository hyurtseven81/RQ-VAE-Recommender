"""Alpha grid search entry point for Stage 2 of the paper plan.

Loads a trained MTL decoder and its RQ-VAE, then iterates through a
per-level alpha grid, evaluating each schedule with
`LevelAwareHybridDecoding` wrapped around `VanillaBeamSearch`. Writes a
CSV with one row per alpha triple and Recall/NDCG@{5,10,20} columns.

Model + tokenizer + dataloader are built once; only the decoding strategy
is swapped per alpha — this is ~N× faster than spawning N separate
`run_eval.py` jobs.

Runs locally or as a SageMaker PyTorch entry_point.

Local usage::

    PYTHONPATH=. python evaluate/alpha_search.py \\
        --gin-config configs/decoder_amazon_beauty_mtl.gin \\
        --decoder-ckpt out/decoder_mtl/amazon_beauty/checkpoint_99999.pt \\
        --rqvae-ckpt   trained_models/rqvae_amazon_beauty/checkpoint_high_entropy.pt \\
        --dataset beauty \\
        --alpha-grid "0.0,0.5,1.0" \\
        --output results/stage2/beauty_pilot.csv

SageMaker (hyperparameters forwarded as CLI flags; see
`sagemaker/launch/launch_alpha_search.py` for the launcher).
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import gin
import torch
from accelerate import Accelerator
from tqdm import tqdm

# noqa: F401 — registering train_decoder_mtl makes its @gin.configurable
# train_mtl visible so configs/decoder_*_mtl.gin parse cleanly here.
import train_decoder_mtl  # noqa: F401
from data.processed import RecDataset
from data.utils import batch_to
from evaluate.metrics import TopKAccumulator
from evaluate.run_eval import _resolve_ckpt
from modules.decoding.level_aware_mix import LevelAwareHybridDecoding
from modules.decoding.vanilla import VanillaBeamSearch
from modules.heads.sasrec_head import SASRecAuxHead
from modules.tokenizer.semids import SemanticIdTokenizer
from train_decoder import _setup_training


def _parse_alpha_grid(spec: str) -> list[float]:
    return [float(x.strip()) for x in spec.split(",") if x.strip()]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Per-level alpha grid search.")
    p.add_argument("--gin-config", "--gin_config", default=None)
    p.add_argument("--decoder-ckpt", "--decoder-checkpoint", "--decoder_checkpoint",
                   required=True, dest="decoder_ckpt")
    p.add_argument("--rqvae-ckpt", "--rqvae-checkpoint", "--rqvae_checkpoint",
                   required=True, dest="rqvae_ckpt")
    p.add_argument("--dataset", default="beauty",
                   choices=["beauty", "sports", "toys", "steam", "ml1m", "ml32m"])
    p.add_argument("--dataset-folder", "--dataset_folder",
                   default="dataset/amazon", dest="dataset_folder")
    p.add_argument("--dataset-split", "--dataset_split",
                   default=None, dest="dataset_split")

    # Grid definition. Two ways to specify:
    # 1) --alpha-grid "0.0,0.5,1.0"  → same grid per level (cartesian)
    # 2) --alpha0-grid, --alpha1-grid, --alpha2-grid  → separate grids (cartesian)
    p.add_argument("--alpha-grid", "--alpha_grid", default=None, dest="alpha_grid",
                   help="Comma-separated grid values shared across every level.")
    p.add_argument("--alpha0-grid", "--alpha0_grid", default=None, dest="alpha0_grid",
                   help="Comma-separated grid for level 0 (overrides --alpha-grid).")
    p.add_argument("--alpha1-grid", "--alpha1_grid", default=None, dest="alpha1_grid")
    p.add_argument("--alpha2-grid", "--alpha2_grid", default=None, dest="alpha2_grid")

    p.add_argument("--batch-size", "--batch_size", type=int, default=256,
                   dest="batch_size")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", required=True,
                   help="CSV path (or directory) for grid-search results.")
    p.add_argument("--job-name", "--job_name", default=None, dest="job_name")
    return p.parse_args()


def _alpha_combinations(args: argparse.Namespace, n_levels: int) -> list[list[float]]:
    if args.alpha0_grid or args.alpha1_grid or args.alpha2_grid:
        if n_levels != 3:
            raise ValueError(
                f"Per-level grids --alpha{{0,1,2}}-grid assume n_levels=3, got {n_levels}"
            )
        g0 = _parse_alpha_grid(args.alpha0_grid or args.alpha_grid or "0.0,0.5,1.0")
        g1 = _parse_alpha_grid(args.alpha1_grid or args.alpha_grid or "0.0,0.5,1.0")
        g2 = _parse_alpha_grid(args.alpha2_grid or args.alpha_grid or "0.0,0.5,1.0")
        return [list(combo) for combo in itertools.product(g0, g1, g2)]
    grid = _parse_alpha_grid(args.alpha_grid or "0.0,0.5,1.0")
    return [list(combo) for combo in itertools.product(grid, repeat=n_levels)]


def _resolve_output_csv(output: str, job_name: str | None) -> Path:
    p = Path(output)
    if p.suffix == ".csv":
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    p.mkdir(parents=True, exist_ok=True)
    stem = job_name or "alpha_grid"
    return p / f"{stem}.csv"


def main() -> None:
    args = _parse_args()
    torch.manual_seed(args.seed)

    if args.gin_config is not None:
        gin.parse_config_file(args.gin_config)

    accelerator = Accelerator()
    device = accelerator.device
    dataset_split = args.dataset_split or args.dataset

    # Architecture kwargs — query gin (train_mtl.* or train.*) so non-Amazon
    # datasets (e.g. ML32M with vae_embed_dim=64) reconstruct correctly.
    from evaluate._arch import get_arch
    arch = get_arch()
    n_levels = arch["vae_n_layers"]

    dataset_enum_map = {
        "beauty": RecDataset.AMAZON, "sports": RecDataset.AMAZON,
        "toys": RecDataset.AMAZON, "steam": RecDataset.STEAM,
        "ml1m": RecDataset.ML_1M, "ml32m": RecDataset.ML_32M,
    }
    dataset_enum = dataset_enum_map.get(args.dataset, RecDataset.AMAZON)

    rqvae_ckpt_local = _resolve_ckpt(args.rqvae_ckpt)
    decoder_ckpt_local = _resolve_ckpt(args.decoder_ckpt)

    setup = _setup_training(
        dataset_folder=args.dataset_folder,
        dataset=dataset_enum,
        force_dataset_process=False,
        dataset_split=dataset_split,
        batch_size=args.batch_size,
        vae_input_dim=arch["vae_input_dim"],
        vae_hidden_dims=arch["vae_hidden_dims"],
        vae_embed_dim=arch["vae_embed_dim"],
        vae_codebook_size=arch["vae_codebook_size"],
        vae_n_layers=arch["vae_n_layers"],
        vae_n_cat_feats=arch["vae_n_cat_feats"],
        vae_codebook_normalize=arch["vae_codebook_normalize"],
        vae_sim_vq=arch["vae_sim_vq"],
        pretrained_rqvae_path=rqvae_ckpt_local,
        t5_d_model=arch["t5_d_model"],
        t5_num_heads=arch["t5_num_heads"],
        t5_d_ff=arch["t5_d_ff"],
        t5_num_layers=arch["t5_num_layers"],
        top_k_for_generation=arch["top_k_for_generation"],
        should_add_sep_token=arch["should_add_sep_token"],
        num_user_bins=None,
        learning_rate=1e-3,
        weight_decay=1e-4,
        accelerator=accelerator,
        train_data_subsample=False,
    )

    model = setup["model"]
    tokenizer: SemanticIdTokenizer = setup["tokenizer"]
    eval_dataloader = setup["eval_dataloader"]

    ckpt = torch.load(decoder_ckpt_local, map_location="cpu")
    model_state = ckpt.get("model", ckpt)
    model.load_state_dict(model_state, strict=False)
    model = model.to(device)
    model.eval()
    print(f"Loaded decoder from {decoder_ckpt_local} (iter={ckpt.get('iter', '?')})")

    # Hard invariant: train-time codebooks (just loaded into model.codebooks)
    # must equal the freshly recomputed eval-time SID table. If not, every
    # held-out target will miss and recall@K will be identically zero.
    from evaluate._invariants import assert_corpus_ids_match, smoke_test_zero_alpha
    assert_corpus_ids_match(model, tokenizer, n_levels)

    aux_head: SASRecAuxHead | None = None
    if "aux_head" in ckpt:
        num_items = len(setup["item_dataset"])
        aux_head = SASRecAuxHead(
            d_model=arch["t5_d_model"],
            d_item=arch["vae_embed_dim"],
            num_items=num_items,
        )
        aux_head.load_state_dict(ckpt["aux_head"])
        aux_head = aux_head.to(device)
        aux_head.eval()
        print("Loaded SASRecAuxHead from checkpoint.")
    else:
        raise RuntimeError(
            "alpha_search requires an MTL decoder checkpoint containing an "
            "aux_head — the level-aware strategies all need the SASRec head."
        )

    codebook_embs = [
        tokenizer.rq_vae.layers[i].embedding.weight.detach().to(device)
        for i in range(n_levels)
    ]

    alpha_grid = _alpha_combinations(args, n_levels)
    print(f"Alpha grid: {len(alpha_grid)} schedule(s)")

    # Bypass-invariant smoke test: α=[0]*n_levels must equal vanilla beam
    # search per the AGENTS.md invariant. If recall is zero across a few
    # batches, the full grid will be too — abort before burning compute.
    smoke_test_zero_alpha(
        model=model,
        tokenizer=tokenizer,
        eval_dataloader=eval_dataloader,
        aux_head=aux_head,
        codebook_embs=codebook_embs,
        device=device,
        n_levels=n_levels,
    )

    # --- Evaluate each alpha schedule ---
    output_csv = _resolve_output_csv(args.output, args.job_name)
    rows: list[dict] = []
    for i, alpha in enumerate(alpha_grid):
        strategy = LevelAwareHybridDecoding(
            base_strategy=VanillaBeamSearch(),
            alpha=alpha,
            aux_head=aux_head,
        )
        acc = TopKAccumulator(ks=[5, 10, 20])
        for batch in tqdm(eval_dataloader, desc=f"α={alpha}", leave=False):
            data = batch_to(batch, device)
            tokenized = tokenizer(data)
            with torch.no_grad():
                generated = model.generate_next_sem_id(
                    tokenized,
                    top_k=True,
                    temperature=1,
                    strategy=strategy,
                    codebook_embs=codebook_embs,
                )
            target = tokenized.sem_ids_fut[:, :n_levels]
            acc.accumulate(generated_ids=generated.sem_ids, target_ids=target)

        metrics = acc.reduce()
        aggregate = {k: v for k, v in metrics.items() if k != "per_user"}
        row = {
            **{f"alpha_{lvl}": alpha[lvl] for lvl in range(n_levels)},
            **{k: float(v) for k, v in aggregate.items()},
        }
        rows.append(row)
        print(f"[{i + 1}/{len(alpha_grid)}] α={alpha}  " +
              "  ".join(f"{k}={v:.4f}" for k, v in aggregate.items()))

    # TopKAccumulator.reduce() emits keys of the form "recall@{k}" / "ndcg@{k}"
    # (see evaluate/metrics.py). Use the literal key — earlier code looked for
    # "recall_at_10" which silently missed the real key and fell back to
    # an alphabetic no-op sort.
    sort_key = None
    for candidate in ("recall@10", "ndcg@10"):
        if rows and candidate in rows[0]:
            sort_key = candidate
            break
    if sort_key is None:
        print("[warn] no recall/ndcg key in reduce() output; leaving rows unsorted.")
    else:
        rows.sort(key=lambda r: r.get(sort_key, 0.0), reverse=True)

    fieldnames = list(rows[0].keys()) if rows else [f"alpha_{i}" for i in range(n_levels)]
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {output_csv}")

    # Also emit a JSON summary for downstream aggregation.
    summary = {
        "dataset": args.dataset,
        "n_alpha_points": len(rows),
        "sort_key": sort_key,
        "best_by_sort_key": rows[0] if rows else None,
        "csv": str(output_csv),
    }
    summary_path = output_csv.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote summary   to {summary_path}")


if __name__ == "__main__":
    main()
