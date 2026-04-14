"""Evaluation entry point for level-aware hybrid decoding.

Runs held-out test evaluation for a trained decoder under any registered
decoding strategy and writes a result JSON compatible with result_store.py
and the paper's collect_results / make_tables pipeline.

Works both locally and as a SageMaker PyTorch entry_point.

Local usage
-----------
python evaluate/run_eval.py \
    --gin-config configs/decoder_amazon_beauty_mtl.gin \
    --decoder-ckpt out/decoder_mtl/amazon_beauty/checkpoint_99999.pt \
    --rqvae-ckpt trained_models/rqvae_amazon_beauty/checkpoint_399999.pt \
    --dataset beauty \
    --strategy level_aware_mix_learned \
    --alpha-ckpt out/alpha_params/beauty_learned.pt \
    --output results/

SageMaker (hyperparameters are forwarded as CLI flags):
    --gin-config / --gin_config
    --decoder-checkpoint / --decoder_checkpoint
    --rqvae-checkpoint   / --rqvae_checkpoint
    --strategy
"""
from __future__ import annotations

import argparse
import json
import os
import time

import gin
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.processed import RecDataset
from data.utils import batch_to
from evaluate.metrics import TopKAccumulator
from evaluate.result_store import write_result
from modules.decoding import DECODING_STRATEGIES
from modules.decoding.level_aware_mix import AlphaParams, LevelAwareHybridDecoding
from modules.decoding.sasrec_reranker import SASRecReranker
from modules.decoding.vanilla import VanillaBeamSearch
from modules.heads.sasrec_head import SASRecAuxHead
from modules.tokenizer.semids import SemanticIdTokenizer
from train_decoder import _setup_training

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run held-out evaluation.")

    # Model architecture — provided via gin config
    p.add_argument("--gin-config", "--gin_config", default=None,
                   help="Path to gin config file (e.g. configs/decoder_amazon_beauty_mtl.gin).")

    # Checkpoints (local paths or S3 URIs when running on SageMaker)
    p.add_argument("--decoder-ckpt", "--decoder-checkpoint", "--decoder_checkpoint",
                   required=True, dest="decoder_ckpt",
                   help="Path to decoder checkpoint .pt file.")
    p.add_argument("--rqvae-ckpt", "--rqvae-checkpoint", "--rqvae_checkpoint",
                   required=True, dest="rqvae_ckpt",
                   help="Path to RQ-VAE checkpoint .pt file.")

    # Dataset
    p.add_argument("--dataset", default="beauty",
                   choices=["beauty", "sports", "toys", "steam", "ml1m", "ml32m"],
                   help="Dataset name.")
    p.add_argument("--dataset-folder", "--dataset_folder", default="dataset/amazon",
                   dest="dataset_folder")
    p.add_argument("--dataset-split", "--dataset_split", default=None, dest="dataset_split",
                   help="Dataset split name (e.g. 'beauty'). Defaults to --dataset value.")

    # Decoding strategy
    p.add_argument("--strategy", default="vanilla",
                   choices=list(DECODING_STRATEGIES.keys()),
                   help="Decoding strategy to evaluate.")

    # Strategy-specific: alpha sources
    p.add_argument("--alpha-ckpt", "--alpha_ckpt", default=None, dest="alpha_ckpt",
                   help="Path to AlphaParams .pt file (for level_aware_mix_learned).")
    p.add_argument("--alpha-csv", "--alpha_csv", default=None, dest="alpha_csv",
                   help="Path to grid search CSV (for level_aware_mix_grid). "
                        "Best alpha by recall@10 is used.")
    p.add_argument("--alpha", default=None,
                   help="Comma-separated alpha values, e.g. '0.5,0.25,0.0' "
                        "(for level_aware_mix). Overrides --alpha-ckpt/--alpha-csv.")

    # SASRec aux head (required for level_aware_mix* and sasrec_rerank)
    p.add_argument("--sasrec-alpha", "--sasrec_alpha", type=float, default=0.5,
                   dest="sasrec_alpha",
                   help="Mixing weight for sasrec_rerank strategy.")

    # Eval settings
    p.add_argument("--batch-size", "--batch_size", type=int, default=256, dest="batch_size")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="results/", help="Output directory for result JSON.")
    p.add_argument("--job-name", "--job_name", default=None, dest="job_name",
                   help="Job name used in result filename. Auto-generated if omitted.")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Strategy instantiation
# ---------------------------------------------------------------------------

def _build_strategy(
    strategy_name: str,
    args: argparse.Namespace,
    aux_head: SASRecAuxHead | None,
    n_levels: int,
) -> object:
    """Instantiate the requested decoding strategy."""

    if strategy_name == "vanilla":
        return VanillaBeamSearch()

    if strategy_name in ("dbs", "gumbel_topk", "hybrid"):
        return DECODING_STRATEGIES[strategy_name]()

    if strategy_name == "sasrec_rerank":
        if aux_head is None:
            raise ValueError("sasrec_rerank requires aux_head (use an MTL checkpoint).")
        return SASRecReranker(aux_head=aux_head, alpha=args.sasrec_alpha)

    if strategy_name in ("level_aware_mix", "level_aware_mix_grid", "level_aware_mix_learned"):
        if aux_head is None:
            raise ValueError(f"{strategy_name} requires aux_head (use an MTL checkpoint).")

        # Resolve alpha schedule
        alpha: list[float]
        if args.alpha is not None:
            alpha = [float(x) for x in args.alpha.split(",")]
        elif strategy_name == "level_aware_mix_learned" or args.alpha_ckpt is not None:
            if args.alpha_ckpt is None:
                raise ValueError("level_aware_mix_learned requires --alpha-ckpt.")
            state = torch.load(args.alpha_ckpt, map_location="cpu")
            params = AlphaParams(n_levels=n_levels)
            params.load_state_dict(state if "phi" in state else state.get("alpha_params", state))
            alpha = params.alpha
        elif strategy_name == "level_aware_mix_grid" or args.alpha_csv is not None:
            if args.alpha_csv is None:
                raise ValueError("level_aware_mix_grid requires --alpha-csv.")
            import pandas as pd
            df = pd.read_csv(args.alpha_csv)
            best = df.sort_values("recall_at_10", ascending=False).iloc[0]
            alpha = [float(best[f"alpha_{i}"]) for i in range(n_levels)]
            print(f"Grid alpha (best recall@10={best['recall_at_10']:.4f}): {alpha}")
        else:
            # level_aware_mix with no explicit alpha — default to 0.5 per level
            alpha = [0.5] * n_levels
            print(f"No alpha source provided for {strategy_name}; using default {alpha}")

        if len(alpha) != n_levels:
            raise ValueError(f"Alpha length {len(alpha)} != n_levels {n_levels}.")

        return LevelAwareHybridDecoding(
            base_strategy=VanillaBeamSearch(),
            alpha=alpha,
            aux_head=aux_head,
        )

    raise ValueError(f"Unknown strategy: {strategy_name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    torch.manual_seed(args.seed)

    # Load gin config if provided (sets model architecture hyperparams)
    if args.gin_config is not None:
        gin.parse_config_file(args.gin_config)

    accelerator = Accelerator()
    device = accelerator.device

    dataset_split = args.dataset_split or args.dataset

    # ------------------------------------------------------------------
    # Build model via shared setup helper (reuses train_decoder.py logic)
    # ------------------------------------------------------------------
    # Fall back to standard Amazon architecture constants when no gin config.
    # These match the default configs/decoder_amazon*.gin values.
    @gin.configurable
    def _get_arch(
        vae_input_dim: int = 768,
        vae_hidden_dims: list = None,
        vae_embed_dim: int = 32,
        vae_n_cat_feats: int = 0,
        vae_codebook_size: int = 256,
        vae_n_layers: int = 3,
        vae_codebook_normalize: bool = False,
        vae_sim_vq: bool = False,
        t5_d_model: int = 384,
        t5_num_heads: int = 6,
        t5_d_ff: int = 1024,
        t5_num_layers: int = 4,
        top_k_for_generation: int = 10,
        should_add_sep_token: bool = True,
        batch_size: int = 256,
    ) -> dict:
        return {
            "vae_input_dim": vae_input_dim,
            "vae_hidden_dims": vae_hidden_dims or [512, 256, 128],
            "vae_embed_dim": vae_embed_dim,
            "vae_n_cat_feats": vae_n_cat_feats,
            "vae_codebook_size": vae_codebook_size,
            "vae_n_layers": vae_n_layers,
            "vae_codebook_normalize": vae_codebook_normalize,
            "vae_sim_vq": vae_sim_vq,
            "t5_d_model": t5_d_model,
            "t5_num_heads": t5_num_heads,
            "t5_d_ff": t5_d_ff,
            "t5_num_layers": t5_num_layers,
            "top_k_for_generation": top_k_for_generation,
            "should_add_sep_token": should_add_sep_token,
            "batch_size": args.batch_size or batch_size,
        }

    arch = _get_arch()

    # Determine dataset enum
    dataset_enum_map = {
        "beauty": RecDataset.AMAZON, "sports": RecDataset.AMAZON,
        "toys": RecDataset.AMAZON, "steam": RecDataset.STEAM,
        "ml1m": RecDataset.ML1M, "ml32m": RecDataset.ML32M,
    }
    dataset_enum = dataset_enum_map.get(args.dataset, RecDataset.AMAZON)

    setup = _setup_training(
        dataset_folder=args.dataset_folder,
        dataset=dataset_enum,
        force_dataset_process=False,
        dataset_split=dataset_split,
        batch_size=arch["batch_size"],
        vae_input_dim=arch["vae_input_dim"],
        vae_hidden_dims=arch["vae_hidden_dims"],
        vae_embed_dim=arch["vae_embed_dim"],
        vae_codebook_size=arch["vae_codebook_size"],
        vae_n_layers=arch["vae_n_layers"],
        vae_n_cat_feats=arch["vae_n_cat_feats"],
        vae_codebook_normalize=arch["vae_codebook_normalize"],
        vae_sim_vq=arch["vae_sim_vq"],
        pretrained_rqvae_path=args.rqvae_ckpt,
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
    eval_dataloader: DataLoader = setup["eval_dataloader"]
    n_levels = arch["vae_n_layers"]

    # ------------------------------------------------------------------
    # Load decoder checkpoint weights
    # ------------------------------------------------------------------
    ckpt = torch.load(args.decoder_ckpt, map_location="cpu")
    model_state = ckpt.get("model", ckpt)
    model.load_state_dict(model_state, strict=False)
    model = model.to(device)
    model.eval()
    print(f"Loaded decoder from {args.decoder_ckpt} (iter={ckpt.get('iter', '?')})")

    # ------------------------------------------------------------------
    # Load aux head (if present in checkpoint)
    # ------------------------------------------------------------------
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

    # Codebook embeddings for embedding-space strategies
    codebook_embs = [
        tokenizer.rq_vae.vq.layers[i].embedding.weight.detach().to(device)
        for i in range(n_levels)
    ]

    # ------------------------------------------------------------------
    # Build decoding strategy
    # ------------------------------------------------------------------
    strategy = _build_strategy(args.strategy, args, aux_head, n_levels)
    print(f"Strategy: {args.strategy} → {type(strategy).__name__}")

    # ------------------------------------------------------------------
    # Evaluation loop
    # ------------------------------------------------------------------
    acc = TopKAccumulator(ks=[1, 5, 10, 20])
    with tqdm(eval_dataloader, desc="Evaluating") as pbar:
        for batch in pbar:
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
            acc.accumulate(
                generated_ids=generated.sem_ids,
                target_ids=target,
            )

    metrics = acc.reduce()
    aggregate = {k: v for k, v in metrics.items() if k != "per_user"}
    print("Results:", json.dumps(aggregate, indent=2))

    # ------------------------------------------------------------------
    # Resolve alpha schedule for result record
    # ------------------------------------------------------------------
    alpha_schedule: list[float] = []
    if isinstance(strategy, LevelAwareHybridDecoding):
        alpha_schedule = strategy.alpha if hasattr(strategy, "alpha") else []

    # ------------------------------------------------------------------
    # Write result
    # ------------------------------------------------------------------
    decoder_type = "mtl" if aux_head is not None else "vanilla"
    job_name = args.job_name or f"{args.dataset}_{args.strategy}_{int(time.time())}"

    result = {
        "job_name": job_name,
        "dataset": args.dataset,
        "decoder_type": decoder_type,
        "decoding_strategy": args.strategy,
        "alpha_schedule": alpha_schedule,
        "seed": args.seed,
        "aggregate": aggregate,
        "per_user": metrics.get("per_user", {}),
    }

    output_path = os.path.join(args.output, f"{job_name}.json")
    write_result(result, output_path)
    print(f"Result saved to {output_path}")


if __name__ == "__main__":
    main()
