"""Learn per-level alpha parameters with frozen decoder.

Freezes the decoder and auxiliary head, then optimizes alpha_params
(via sigmoid) for 1 epoch to minimize InfoNCE loss on the training set.

Alpha params are initialized to 0 (sigmoid(0)=0.5) and optimized end-to-end.
The decoder and aux_head gradients are disabled; only AlphaParams.phi is learned.

Usage:
    python scripts/train_alpha_params.py \
        --checkpoint out/decoder_mtl/amazon_beauty/best.pt \
        --dataset beauty \
        --output out/alpha_params/beauty_learned.pt
"""
import argparse
import os

import torch
import torch.nn.functional as F

from modules.decoding.level_aware_mix import AlphaParams


def infonce_loss(
    query: torch.Tensor,        # [B, d_item]
    pos_emb: torch.Tensor,      # [B, d_item]
    neg_embs: torch.Tensor,     # [B, n_neg, d_item]
    temperature: float = 0.1,
) -> torch.Tensor:
    """InfoNCE contrastive loss.

    Args:
        query: User query vectors from aux_head.
        pos_emb: Positive item embedding (ground-truth target).
        neg_embs: Negative item embeddings (in-batch or sampled negatives).
        temperature: Softmax temperature for contrastive scoring.

    Returns:
        Scalar InfoNCE loss.
    """
    B = query.size(0)
    # Positive scores: [B]
    pos_scores = (query * pos_emb).sum(-1) / temperature  # [B]

    # Negative scores: [B, n_neg]
    neg_scores = (query.unsqueeze(1) * neg_embs).sum(-1) / temperature  # [B, n_neg]

    # Concatenate positive and negatives: [B, 1 + n_neg]
    logits = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)

    # Labels: positive is index 0
    labels = torch.zeros(B, dtype=torch.long, device=query.device)
    return F.cross_entropy(logits, labels)


def train_one_epoch(
    model,
    aux_head: torch.nn.Module,
    alpha_params: AlphaParams,
    train_loader,
    optimizer: torch.optim.Optimizer,
    codebook_embs: list[torch.Tensor],
    device: str,
) -> float:
    """Train alpha_params for one epoch using InfoNCE loss.

    The decoder and aux_head are frozen; only alpha_params.phi is updated.

    Args:
        model: Decoder model (frozen).
        aux_head: SASRecAuxHead (frozen).
        alpha_params: AlphaParams module with learnable phi.
        train_loader: DataLoader yielding (user_seq, target_item_emb, neg_item_embs).
        optimizer: Optimizer for alpha_params only.
        codebook_embs: Codebook embedding tables.
        device: torch device string.

    Returns:
        Mean InfoNCE loss over the epoch.
    """

    alpha_params.train()
    total_loss = 0.0
    n_batches = 0

    for batch in train_loader:
        user_seqs, target_embs, neg_embs = batch
        user_seqs = user_seqs.to(device)
        target_embs = target_embs.to(device)   # [B, d_item]
        neg_embs = neg_embs.to(device)          # [B, n_neg, d_item]

        optimizer.zero_grad()

        # Forward pass through frozen decoder to get hidden states
        with torch.no_grad():
            decoder_hidden = model.encode_user(user_seqs)  # [B, d_model]

        # Compute query from frozen aux_head
        with torch.no_grad():
            q = aux_head(decoder_hidden)  # [B, d_item]

        # Scale query by alpha_params: weighted sum of autoregressive + dense signal
        # Use current alpha schedule as mixing weights for contrastive loss
        # Use mean alpha as a scalar proxy for the overall mixing weight
        # This trains phi to maximize InfoNCE between q and target embeddings
        alpha_mean = torch.sigmoid(alpha_params.phi).mean()  # differentiable

        # Scale query by alpha (differentiable wrt phi)
        q_scaled = q * alpha_mean

        loss = infonce_loss(q_scaled, target_embs, neg_embs)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches if n_batches > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(
        description="Learn per-level alpha parameters with frozen decoder and aux_head."
    )
    parser.add_argument(
        "--checkpoint", required=True, help="Path to MTL decoder checkpoint (.pt)"
    )
    parser.add_argument(
        "--dataset",
        choices=["beauty", "sports", "toys", "steam"],
        required=True,
        help="Dataset name for loading training data.",
    )
    parser.add_argument(
        "--output", required=True, help="Output path for learned alpha params (.pt)"
    )
    parser.add_argument("--n-levels", type=int, default=3, help="Number of RQ-VAE codebook levels.")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate for alpha params.")
    parser.add_argument("--n-epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size.")
    parser.add_argument("--device", default="cuda", help="Device string (cuda or cpu).")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    if device != args.device:
        print(f"Warning: {args.device} not available, falling back to cpu.")

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    model = checkpoint.get("model") or checkpoint.get("model_state_dict")
    aux_head_state = checkpoint.get("aux_head") or checkpoint.get("aux_head_state_dict")
    codebook_embs = checkpoint.get("codebook_embs")

    if model is None or aux_head_state is None:
        raise ValueError(
            "Checkpoint must contain 'model' (or 'model_state_dict') and "
            "'aux_head' (or 'aux_head_state_dict') keys."
        )
    if codebook_embs is None:
        raise ValueError("Checkpoint must contain 'codebook_embs' key.")

    model = model.to(device)
    aux_head = aux_head_state.to(device) if isinstance(aux_head_state, torch.nn.Module) else None
    if aux_head is None:
        raise ValueError("Could not load aux_head from checkpoint.")
    codebook_embs = [e.to(device) for e in codebook_embs]

    # Freeze decoder and aux_head
    for param in model.parameters():
        param.requires_grad_(False)
    for param in aux_head.parameters():
        param.requires_grad_(False)
    model.eval()
    aux_head.eval()

    # Initialize learnable alpha params (phi=0 → alpha=0.5)
    alpha_params = AlphaParams(n_levels=args.n_levels).to(device)
    print(f"Initial alpha schedule: {alpha_params.alpha}")

    # Optimizer only on alpha_params
    optimizer = torch.optim.Adam(alpha_params.parameters(), lr=args.lr)

    # Load training data
    from torch.utils.data import DataLoader

    from data.processed import RecDataset as RecDS
    from data.processed import SeqData

    dataset_enum_map = {
        "beauty": RecDS.AMAZON, "sports": RecDS.AMAZON,
        "toys": RecDS.AMAZON, "steam": RecDS.STEAM,
    }
    dataset_folder_map = {
        "beauty": "dataset/amazon", "sports": "dataset/amazon",
        "toys": "dataset/amazon", "steam": "dataset/steam",
    }
    train_ds = SeqData(
        root=dataset_folder_map[args.dataset],
        dataset=dataset_enum_map[args.dataset],
        is_train=True,
        subsample=False,
        split=args.dataset,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    # Training loop
    for epoch in range(1, args.n_epochs + 1):
        mean_loss = train_one_epoch(
            model=model,
            aux_head=aux_head,
            alpha_params=alpha_params,
            train_loader=train_loader,
            optimizer=optimizer,
            codebook_embs=codebook_embs,
            device=device,
        )
        print(
            f"Epoch {epoch}/{args.n_epochs} — InfoNCE loss: {mean_loss:.4f} | "
            f"alpha: {[round(a, 4) for a in alpha_params.alpha]}"
        )

    # Save learned alpha params
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(
        {
            "alpha_params_state_dict": alpha_params.state_dict(),
            "alpha": alpha_params.alpha,
            "n_levels": args.n_levels,
            "dataset": args.dataset,
            "source_checkpoint": args.checkpoint,
        },
        args.output,
    )
    print(f"\nLearned alpha params saved to {args.output}")
    print(f"Final alpha schedule: {alpha_params.alpha}")


if __name__ == "__main__":
    main()
