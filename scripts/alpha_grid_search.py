"""Coarse grid search for per-level alpha schedule.

Evaluates all alpha combinations on the validation set.
With L=3 levels and grid {0, 0.25, 0.5, 0.75, 1.0}: 5^3 = 125 configs.

Usage:
    python scripts/alpha_grid_search.py \
        --checkpoint out/decoder_mtl/amazon_beauty/best.pt \
        --dataset beauty \
        --output results/alpha_search/beauty_grid.csv
"""
import argparse
import itertools
import os
import torch
import pandas as pd
from typing import List, Tuple

GRID_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0]


def evaluate_alpha_schedule(
    model,
    aux_head,
    val_loader,
    alpha_schedule: List[float],
    codebook_embs: List[torch.Tensor],
    device: str,
) -> float:
    """Run inference with given alpha schedule, return Recall@10.

    Args:
        model: Decoder model with a generate() method.
        aux_head: Trained SASRecAuxHead (frozen).
        val_loader: DataLoader yielding (user_seq, target_item) batches.
        alpha_schedule: Per-level alpha values, e.g. [0.5, 0.25, 0.0].
        codebook_embs: List of [vocab, d_item] codebook embedding tables.
        device: torch device string.

    Returns:
        Recall@10 on the validation set (float in [0, 1]).
    """
    from modules.decoding.level_aware_mix import LevelAwareHybridDecoding
    from modules.decoding.vanilla import VanillaBeamSearch

    strategy = LevelAwareHybridDecoding(
        base_strategy=VanillaBeamSearch(),
        alpha=alpha_schedule,
        aux_head=aux_head,
    )

    model.eval()
    aux_head.eval()
    hits = 0
    total = 0

    with torch.no_grad():
        for batch in val_loader:
            user_seqs, target_ids = batch
            user_seqs = user_seqs.to(device)
            target_ids = target_ids.to(device)  # [B]

            # generate() should accept a decoding_strategy kwarg
            beams, _ = model.generate(
                input_ids=user_seqs,
                k=10,
                decoding_strategy=strategy,
                codebook_embs=codebook_embs,
            )
            # beams: [B, k, L] — check if target appears in top-10
            B = target_ids.size(0)
            for b in range(B):
                target_sid = target_ids[b]  # scalar SID index
                # Check if any beam matches target
                # This assumes SIDs are stored as multi-hot tuples matched against corpus
                # In practice, model.generate() returns SID tuples; comparison is dataset-specific
                # For now: check if target_sid appears in the first k beams
                found = (beams[b] == target_sid.unsqueeze(0)).all(dim=-1).any()
                hits += int(found.item())
            total += B

    return hits / total if total > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(
        description="Grid search over per-level alpha schedules for level-aware decoding."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to decoder checkpoint (.pt)")
    parser.add_argument(
        "--dataset",
        choices=["beauty", "sports", "toys", "steam"],
        required=True,
        help="Dataset name for loading configs and data.",
    )
    parser.add_argument("--n-levels", type=int, default=3, help="Number of RQ-VAE codebook levels.")
    parser.add_argument(
        "--output", required=True, help="Output CSV path for grid search results."
    )
    parser.add_argument("--device", default="cuda", help="Device string (cuda or cpu).")
    parser.add_argument("--batch-size", type=int, default=64, help="Inference batch size.")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    if device != args.device:
        print(f"Warning: {args.device} not available, falling back to cpu.")

    # Build all alpha combinations
    all_alphas = list(itertools.product(GRID_VALUES, repeat=args.n_levels))
    print(f"Evaluating {len(all_alphas)} alpha configurations on {args.dataset}...")

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Extract components from checkpoint
    # Expected checkpoint format: {"model": state_dict, "aux_head": state_dict, ...}
    # Adjust keys to match actual checkpoint format.
    from modules.model import TIGER  # noqa: F401 — imported for type reference
    from modules.heads.sasrec_head import SASRecAuxHead

    model = checkpoint.get("model") or checkpoint.get("model_state_dict")
    aux_head_state = checkpoint.get("aux_head") or checkpoint.get("aux_head_state_dict")

    if model is None or aux_head_state is None:
        raise ValueError(
            "Checkpoint must contain 'model' (or 'model_state_dict') and "
            "'aux_head' (or 'aux_head_state_dict') keys."
        )

    # If state dicts were stored, the caller should reconstruct the model.
    # For simplicity, assume checkpoint stores full model objects.
    model = model.to(device)
    aux_head = aux_head_state.to(device) if isinstance(aux_head_state, torch.nn.Module) else None

    if aux_head is None:
        raise ValueError("Could not load aux_head from checkpoint.")

    # Load codebook embeddings from checkpoint
    codebook_embs = checkpoint.get("codebook_embs")
    if codebook_embs is None:
        raise ValueError("Checkpoint must contain 'codebook_embs' key.")
    codebook_embs = [e.to(device) for e in codebook_embs]

    # Load dataset
    from data.dataset import get_val_loader  # adjust to actual data module path

    val_loader = get_val_loader(args.dataset, batch_size=args.batch_size)

    # Run grid search
    results = []
    for i, alpha_tuple in enumerate(all_alphas):
        alpha_schedule = list(alpha_tuple)
        recall = evaluate_alpha_schedule(
            model=model,
            aux_head=aux_head,
            val_loader=val_loader,
            alpha_schedule=alpha_schedule,
            codebook_embs=codebook_embs,
            device=device,
        )
        results.append({"alpha": str(alpha_schedule), **{f"alpha_{l}": alpha_tuple[l] for l in range(args.n_levels)}, "recall_at_10": recall})

        if (i + 1) % 10 == 0 or (i + 1) == len(all_alphas):
            print(f"  [{i+1}/{len(all_alphas)}] alpha={alpha_schedule} → Recall@10={recall:.4f}")

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df = pd.DataFrame(results).sort_values("recall_at_10", ascending=False)
    df.to_csv(args.output, index=False)
    print(f"\nGrid search complete. Best config:")
    print(df.head(5).to_string(index=False))
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
