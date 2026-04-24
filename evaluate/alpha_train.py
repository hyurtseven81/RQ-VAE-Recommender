"""Learn per-level alpha parameters with a frozen MTL decoder (Stage 2 — learned alpha).

Loads a trained MTL decoder + its RQ-VAE, freezes both, and optimises an
`AlphaParams` module by minimising the teacher-forced mixed cross-entropy
loss:

    for each level h:
        log_p_h = decoder_mlp[h](decoder_hidden_h)     # [B, vocab]
        q_h     = aux_head(decoder_hidden_h)            # [B, d_item]
        r_h     = sum_{i<h} e_{c*_i}[0:d_item] + codebook_h   # [B, vocab, d_item]
        s_dense = <q_h, r_h>                             # [B, vocab]
        alpha_h = sigmoid(phi[h])
        mixed   = _mix_logspace(log_p_h, s_dense, alpha_h)   # same op as inference
        loss_h  = cross_entropy(mixed, fut_ids[:, h])

    loss = mean_h loss_h
    backward only through phi

The mixing function is imported from modules.decoding.level_aware_mix so
the training-time and inference-time mixing are byte-for-byte the same.

Writes a .pt file with the learned `AlphaParams.state_dict()`, the final
alpha triple, and a small JSON summary suitable for the paper table.

Runs locally or as a SageMaker PyTorch entry_point.

Local usage::

    PYTHONPATH=. python evaluate/alpha_train.py \\
        --gin-config configs/decoder_amazon_beauty_mtl.gin \\
        --decoder-ckpt out/decoder_mtl/amazon_beauty/checkpoint_99999.pt \\
        --rqvae-ckpt   trained_models/rqvae_amazon_beauty/checkpoint_high_entropy.pt \\
        --dataset beauty \\
        --output out/alpha_params/beauty_learned.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import gin
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.processed import RecDataset
from data.utils import batch_to
from evaluate.run_eval import _resolve_ckpt
from modules.decoding.level_aware_mix import AlphaParams, LevelAwareHybridDecoding
from modules.heads.sasrec_head import SASRecAuxHead
from modules.tokenizer.semids import SemanticIdTokenizer
from train_decoder import _setup_training


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Learn per-level alpha parameters.")
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
    p.add_argument("--batch-size", "--batch_size", type=int, default=256,
                   dest="batch_size")
    p.add_argument("--lr", type=float, default=0.05,
                   help="Learning rate for the alpha phi parameters.")
    p.add_argument("--n-epochs", "--n_epochs", type=int, default=1, dest="n_epochs")
    p.add_argument("--max-steps", "--max_steps", type=int, default=0,
                   dest="max_steps",
                   help="If > 0, cap the number of training steps (useful for "
                        "quick convergence checks; 0 means full epoch(s)).")
    p.add_argument("--init-alpha", default=None,
                   help="Comma-separated initial alpha values (e.g. '0.5,0.5,0.5'). "
                        "If omitted, alphas initialise to sigmoid(0)=0.5.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", required=True,
                   help="Output .pt path (learned AlphaParams state dict + metadata).")
    p.add_argument("--job-name", "--job_name", default=None, dest="job_name")
    return p.parse_args()


def _init_phi_from_alpha(alpha_list: list[float]) -> torch.Tensor:
    """Return phi such that sigmoid(phi) ≈ alpha. Clamp alpha to avoid ±inf."""
    alpha = torch.tensor(alpha_list, dtype=torch.float32).clamp(1e-4, 1 - 1e-4)
    return torch.log(alpha / (1 - alpha))


@torch.no_grad()
def _prepare_teacher_forced_hidden(
    model,
    batch,
    device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the MTL decoder in teacher-forced mode and return
    (decoder_hidden [B, L, d_model], fut_ids [B, L]).
    """
    from modules.model import _strip_dedup_col  # local import, cheap

    sem_ids_dim = model.num_hierarchies + 1
    input_ids = _strip_dedup_col(batch.sem_ids, sem_ids_dim, model.num_hierarchies)
    attention_mask = _strip_dedup_col(
        batch.seq_mask.long(), sem_ids_dim, model.num_hierarchies
    )
    fut_ids = batch.sem_ids_fut[:, : model.num_hierarchies]

    enc_out, enc_mask = model.encoder_forward_pass(
        attention_mask=attention_mask,
        input_ids=input_ids,
        user_id=batch.user_ids,
    )
    dec_out = model.decoder_forward_pass(
        future_ids=fut_ids,
        encoder_output=enc_out,
        attention_mask_for_encoder=enc_mask,
        use_cache=False,
    )[:, :-1]  # [B, L, d_model]
    return dec_out, fut_ids


def _per_level_mixed_loss(
    model,
    aux_head: SASRecAuxHead,
    alpha_params: AlphaParams,
    decoder_hidden: torch.Tensor,   # [B, L, d_model]
    fut_ids: torch.Tensor,          # [B, L]
    codebook_embs: list[torch.Tensor],   # per-level [vocab, d_item]
) -> torch.Tensor:
    """Teacher-forced mixed-logit cross-entropy, summed over levels.

    The mixing step mirrors ``LevelAwareHybridDecoding._mix_logspace`` so
    that training-time and inference-time mixing are identical.
    """
    mix_fn = LevelAwareHybridDecoding(
        base_strategy=None,  # unused here; we only call _mix_logspace / _z_score
        alpha=[0.0] * model.num_hierarchies,
        aux_head=aux_head,
    )

    B, L, _ = decoder_hidden.shape
    loss = decoder_hidden.new_zeros(())
    for h in range(L):
        hidden_h = decoder_hidden[:, h]                       # [B, d_model]
        log_p = torch.log_softmax(
            model.decoder_mlp[h](hidden_h), dim=-1
        )                                                     # [B, vocab]

        # Partial reconstruction: e_{c*_0} + ... + e_{c*_{h-1}} + e_cand_h.
        # Uses teacher-forced ground-truth prefix so the training target is
        # well-defined for every level.
        with torch.no_grad():
            cand_embs = codebook_embs[h]                      # [vocab, d_item]
            if h == 0:
                r = cand_embs.unsqueeze(0).expand(B, -1, -1)  # [B, vocab, d_item]
            else:
                past = sum(
                    codebook_embs[i][fut_ids[:, i]]
                    for i in range(h)
                )                                             # [B, d_item]
                r = past.unsqueeze(1) + cand_embs.unsqueeze(0)  # [B, vocab, d_item]
            q = aux_head(hidden_h)                            # [B, d_item]
            s_dense = (q.unsqueeze(1) * r).sum(-1)            # [B, vocab]

        # sigmoid(phi[h]) is differentiable; everything else is detached.
        alpha_h = torch.sigmoid(alpha_params.phi[h])
        norm_log_p = mix_fn._z_score(log_p.detach(), dim=-1)
        norm_s = mix_fn._z_score(s_dense, dim=-1)
        mixed_z = (1.0 - alpha_h) * norm_log_p + alpha_h * norm_s
        log_p_std = log_p.detach().std(dim=-1, keepdim=True)
        mixed = mixed_z * log_p_std                           # [B, vocab]

        loss_h = torch.nn.functional.cross_entropy(mixed, fut_ids[:, h].long())
        loss = loss + loss_h

    return loss / L


def main() -> None:
    args = _parse_args()
    torch.manual_seed(args.seed)

    if args.gin_config is not None:
        gin.parse_config_file(args.gin_config)

    accelerator = Accelerator()
    device = accelerator.device
    dataset_split = args.dataset_split or args.dataset

    @gin.configurable
    def _arch(
        vae_input_dim: int = 768,
        vae_hidden_dims: list | None = None,
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
    ) -> dict:
        return dict(
            vae_input_dim=vae_input_dim,
            vae_hidden_dims=vae_hidden_dims or [512, 256, 128],
            vae_embed_dim=vae_embed_dim,
            vae_n_cat_feats=vae_n_cat_feats,
            vae_codebook_size=vae_codebook_size,
            vae_n_layers=vae_n_layers,
            vae_codebook_normalize=vae_codebook_normalize,
            vae_sim_vq=vae_sim_vq,
            t5_d_model=t5_d_model,
            t5_num_heads=t5_num_heads,
            t5_d_ff=t5_d_ff,
            t5_num_layers=t5_num_layers,
            top_k_for_generation=top_k_for_generation,
            should_add_sep_token=should_add_sep_token,
        )

    arch = _arch()

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

    wrapped_model = setup["model"]
    tokenizer: SemanticIdTokenizer = setup["tokenizer"]
    train_dataloader: DataLoader = setup["train_dataloader"]

    # _setup_training wraps the model with accelerator.prepare; unwrap so we can
    # call the private encoder/decoder forward helpers directly in the
    # teacher-forced loss computation.
    model = accelerator.unwrap_model(wrapped_model)
    tokenizer = accelerator.unwrap_model(tokenizer)

    ckpt = torch.load(decoder_ckpt_local, map_location="cpu")
    model_state = ckpt.get("model", ckpt)
    model.load_state_dict(model_state, strict=False)
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"Loaded decoder from {decoder_ckpt_local} (iter={ckpt.get('iter', '?')})")

    if "aux_head" not in ckpt:
        raise RuntimeError(
            "alpha_train requires an MTL decoder checkpoint that contains an "
            "aux_head — the learned-alpha objective needs the SASRec head."
        )
    num_items = len(setup["item_dataset"])
    aux_head = SASRecAuxHead(
        d_model=arch["t5_d_model"],
        d_item=arch["vae_embed_dim"],
        num_items=num_items,
    )
    aux_head.load_state_dict(ckpt["aux_head"])
    aux_head = aux_head.to(device)
    aux_head.eval()
    for p in aux_head.parameters():
        p.requires_grad_(False)

    codebook_embs = [
        tokenizer.rq_vae.layers[i].embedding.weight.detach().to(device)
        for i in range(arch["vae_n_layers"])
    ]

    alpha_params = AlphaParams(n_levels=arch["vae_n_layers"]).to(device)
    if args.init_alpha:
        init_list = [float(x) for x in args.init_alpha.split(",")]
        if len(init_list) != arch["vae_n_layers"]:
            raise ValueError(
                f"--init-alpha length {len(init_list)} != n_levels "
                f"{arch['vae_n_layers']}"
            )
        with torch.no_grad():
            alpha_params.phi.copy_(_init_phi_from_alpha(init_list).to(device))
    print(f"Initial alpha: {alpha_params.alpha}")

    optimizer = torch.optim.Adam(alpha_params.parameters(), lr=args.lr)

    alpha_history: list[list[float]] = [list(alpha_params.alpha)]
    for epoch in range(1, args.n_epochs + 1):
        total_loss = 0.0
        n_steps = 0
        for step, batch in enumerate(
            tqdm(train_dataloader, desc=f"Epoch {epoch}", leave=False)
        ):
            if args.max_steps and step >= args.max_steps:
                break
            data = batch_to(batch, device)
            tokenized = tokenizer(data)
            dec_hidden, fut_ids = _prepare_teacher_forced_hidden(model, tokenized, device)

            optimizer.zero_grad(set_to_none=True)
            loss = _per_level_mixed_loss(
                model, aux_head, alpha_params,
                decoder_hidden=dec_hidden,
                fut_ids=fut_ids,
                codebook_embs=codebook_embs,
            )
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            n_steps += 1

        mean_loss = total_loss / max(n_steps, 1)
        print(
            f"Epoch {epoch}/{args.n_epochs} — mean mixed CE: {mean_loss:.4f} "
            f"| alpha: {[round(a, 4) for a in alpha_params.alpha]}"
        )
        alpha_history.append(list(alpha_params.alpha))

    # --- Persist ---
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "alpha_params_state_dict": alpha_params.state_dict(),
            "phi": alpha_params.phi.detach().cpu(),
            "alpha": alpha_params.alpha,
            "n_levels": arch["vae_n_layers"],
            "dataset": args.dataset,
            "source_decoder_ckpt": args.decoder_ckpt,
            "source_rqvae_ckpt": args.rqvae_ckpt,
        },
        output_path,
    )
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps({
        "dataset": args.dataset,
        "final_alpha": alpha_params.alpha,
        "alpha_history": alpha_history,
        "n_epochs": args.n_epochs,
        "n_steps_per_epoch_cap": args.max_steps or None,
        "lr": args.lr,
    }, indent=2))
    print(f"\nLearned alpha saved to {output_path}")
    print(f"Summary        at {summary_path}")
    print(f"Final alpha:     {alpha_params.alpha}")


if __name__ == "__main__":
    main()
