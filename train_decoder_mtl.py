"""Multi-task learning decoder training: SID cross-entropy + SASRec InfoNCE.

Key differences from train_decoder.py:
1. Builds a SASRecAuxHead at init, initialized from RQ-VAE encoder outputs.
2. Aux head parameters are added to the AdamW optimizer.
3. Forward pass extracts decoder_hidden = decoder_output[:, -1, :] (shape [B, 384])
   after teacher-forcing through all L SID tokens. The model.forward() slices
   decoder_output as [:, :-1], so shape is [B, L, 384]; index -1 gives [B, 384].
4. Aux loss: sampled-softmax InfoNCE against the SASRec item embedding table.
5. Total loss: L_sid + lambda_warmup(step) * L_sasrec.
6. Gradient clipping is ALWAYS applied (clip_grad_norm_); vanilla train() only clips
   when max_grad_norm is not None.
7. Checkpoints save aux_head state dict alongside the main model.
"""

import os
import gin
import torch
import torch.nn.functional as F
import wandb

from accelerate import Accelerator
from data.utils import batch_to
from data.utils import cycle
from data.utils import next_batch
from data.processed import RecDataset
from evaluate.metrics import TopKAccumulator
from modules.model import EncoderDecoderRetrievalModel
from modules.heads.sasrec_head import SASRecAuxHead, build_item_embeddings_from_rqvae
from modules.heads.mtl_losses import sasrec_infonce_loss, LambdaWarmupSchedule
from modules.scheduler.inv_sqrt import InverseSquareRootScheduler
from modules.utils import compute_debug_metrics
from modules.utils import override_save_dir_for_sagemaker
from modules.utils import parse_config
from torch.optim import AdamW
from tqdm import tqdm
from typing import Optional

# Import the shared setup helper from train_decoder
from train_decoder import _setup_training


@gin.configurable
def train_mtl(
    # MTL-specific hyper-params
    lambda_aux_max: float = 0.2,
    lambda_aux_warmup_fraction: float = 0.1,
    sasrec_n_negatives: int = 1024,
    sasrec_temperature: float = 0.05,
    grad_clip_norm: float = 1.0,
    # Training params (mirrors train() in train_decoder.py)
    iterations: int = 100000,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 640,
    vae_input_dim: int = 768,
    vae_hidden_dims: list = None,
    vae_embed_dim: int = 32,
    vae_n_cat_feats: int = 0,
    vae_codebook_size: int = 256,
    vae_n_layers: int = 3,
    vae_codebook_normalize: bool = False,
    vae_sim_vq: bool = False,
    pretrained_rqvae_path: str = "",
    pretrained_decoder_path: Optional[str] = None,
    save_dir_root: str = "out/decoder_mtl/",
    dataset_folder: str = "",
    dataset=RecDataset.AMAZON,
    force_dataset_process: bool = False,
    full_eval_every: int = 1000,
    partial_eval_every: int = 5000,
    dataset_split: str = "beauty",
    t5_d_model: int = 384,
    t5_num_heads: int = 6,
    t5_d_ff: int = 1024,
    t5_num_layers: int = 4,
    top_k_for_generation: int = 10,
    should_add_sep_token: bool = True,
    num_user_bins: Optional[int] = None,
    train_data_subsample: bool = True,
    split_batches: bool = True,
    amp: bool = False,
    mixed_precision_type: str = "fp16",
    gradient_accumulate_every: int = 1,
    save_model_every: int = 1000000,
    wandb_logging: bool = False,
    top_k_eval_list: list = None,
):
    """MTL training: SID cross-entropy + SASRec InfoNCE auxiliary loss."""
    if vae_hidden_dims is None:
        vae_hidden_dims = [512, 256, 128]
    if top_k_eval_list is None:
        top_k_eval_list = [1, 5, 10]

    if dataset not in (RecDataset.AMAZON, RecDataset.STEAM, RecDataset.ML_32M, RecDataset.ML_1M):
        raise Exception(f"Dataset currently not supported: {dataset}.")

    if wandb_logging:
        params = locals()

    accelerator = Accelerator(
        split_batches=split_batches,
        mixed_precision=mixed_precision_type if amp else "no",
    )

    if wandb_logging and accelerator.is_main_process:
        wandb.login()
        wandb.init(project="gen-retrieval-decoder-mtl-training", config=params)

    # ------------------------------------------------------------------ #
    # Shared setup: datasets, tokenizer, base model, optimizer, scheduler #
    # ------------------------------------------------------------------ #
    setup = _setup_training(
        dataset_folder=dataset_folder,
        dataset=dataset,
        force_dataset_process=force_dataset_process,
        dataset_split=dataset_split,
        batch_size=batch_size,
        vae_input_dim=vae_input_dim,
        vae_hidden_dims=vae_hidden_dims,
        vae_embed_dim=vae_embed_dim,
        vae_codebook_size=vae_codebook_size,
        vae_n_layers=vae_n_layers,
        vae_n_cat_feats=vae_n_cat_feats,
        vae_codebook_normalize=vae_codebook_normalize,
        vae_sim_vq=vae_sim_vq,
        pretrained_rqvae_path=pretrained_rqvae_path,
        t5_d_model=t5_d_model,
        t5_num_heads=t5_num_heads,
        t5_d_ff=t5_d_ff,
        t5_num_layers=t5_num_layers,
        top_k_for_generation=top_k_for_generation,
        should_add_sep_token=should_add_sep_token,
        num_user_bins=num_user_bins,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        accelerator=accelerator,
        train_data_subsample=train_data_subsample,
        pretrained_decoder_path=pretrained_decoder_path,
    )

    train_dataloader = setup["train_dataloader"]
    eval_dataloader = setup["eval_dataloader"]
    tokenizer = setup["tokenizer"]
    model = setup["model"]
    optimizer = setup["optimizer"]
    lr_scheduler = setup["lr_scheduler"]
    start_iter = setup["start_iter"]
    device = setup["device"]

    # -------------------------------------------------- #
    # Build SASRec aux head from RQ-VAE encoder outputs  #
    # -------------------------------------------------- #
    item_dataset = setup["item_dataset"]
    # item_dataset.item_data is [num_items, feature_dim] — raw item features
    item_features = item_dataset.item_data.float()
    # Truncate to vae_input_dim if features have extra dims (e.g. ML1M genre features)
    if item_features.shape[-1] > vae_input_dim:
        item_features = item_features[..., :vae_input_dim]
    num_items = len(item_features)

    # Use the unwrapped rqvae from the tokenizer (already loaded weights)
    rqvae = accelerator.unwrap_model(tokenizer).rq_vae

    item_embeddings_init = build_item_embeddings_from_rqvae(
        rqvae=rqvae,
        item_features=item_features,
        batch_size=256,
        device=str(device),
    )  # [num_items, vae_embed_dim]

    aux_head = SASRecAuxHead(
        d_model=t5_d_model,
        d_item=vae_embed_dim,
        num_items=num_items,
        dropout=0.1,
        item_embeddings_init=item_embeddings_init,
    ).to(device)

    # Add aux head parameters to the existing optimizer
    optimizer.add_param_group({"params": aux_head.parameters()})
    aux_head = accelerator.prepare(aux_head)

    # ----------------------------------------- #
    # Lambda warmup schedule for aux loss weight #
    # ----------------------------------------- #
    warmup_steps = int(iterations * lambda_aux_warmup_fraction)
    lambda_schedule = LambdaWarmupSchedule(
        lambda_max=lambda_aux_max,
        warmup_steps=warmup_steps,
    )

    metrics_accumulator = TopKAccumulator(ks=top_k_eval_list)
    num_params = sum(p.numel() for p in model.parameters()) + sum(
        p.numel() for p in aux_head.parameters()
    )
    print(f"Device: {device}, Num Parameters (model + aux head): {num_params}")

    # --------------------------------------------------------- #
    # Training loop                                              #
    # --------------------------------------------------------- #
    with tqdm(
        initial=start_iter,
        total=start_iter + iterations,
        disable=not accelerator.is_main_process,
    ) as pbar:
        for iter in range(iterations):
            model.train()
            aux_head.train()
            total_loss = 0.0
            total_sid_loss = 0.0
            total_aux_loss = 0.0
            optimizer.zero_grad()
            train_debug_metrics = {}

            current_lambda = lambda_schedule.get(start_iter + iter)

            for _ in range(gradient_accumulate_every):
                data = next_batch(train_dataloader, device)
                tokenized_data = tokenizer(data)

                with accelerator.autocast():
                    # ---- SID cross-entropy (standard TIGER loss) ---- #
                    # We need the raw decoder output for the aux head, so
                    # we call encoder + decoder manually instead of model.forward().
                    sem_ids_dim = vae_n_layers + 1
                    from modules.model import _strip_dedup_col
                    input_ids = _strip_dedup_col(
                        tokenized_data.sem_ids, sem_ids_dim, vae_n_layers
                    )
                    attention_mask = _strip_dedup_col(
                        tokenized_data.seq_mask.long(), sem_ids_dim, vae_n_layers
                    )
                    fut_ids = tokenized_data.sem_ids_fut[:, :vae_n_layers]

                    unwrapped_model = accelerator.unwrap_model(model)
                    encoder_output, attention_mask_for_encoder = (
                        unwrapped_model.encoder_forward_pass(
                            attention_mask=attention_mask,
                            input_ids=input_ids,
                            user_id=tokenized_data.user_ids,
                        )
                    )
                    # decoder_output: [B, vae_n_layers + 1, t5_d_model]
                    # After [:, :-1]: [B, vae_n_layers, t5_d_model]
                    decoder_output_full = unwrapped_model.decoder_forward_pass(
                        future_ids=fut_ids,
                        encoder_output=encoder_output,
                        attention_mask_for_encoder=attention_mask_for_encoder,
                        use_cache=False,
                    )[:, :-1]  # [B, vae_n_layers, t5_d_model]

                    # SID loss: per-hierarchy cross-entropy
                    sid_loss = torch.tensor(0.0, device=device)
                    for h in range(vae_n_layers):
                        logits = unwrapped_model.decoder_mlp[h](decoder_output_full[:, h])
                        sid_loss = sid_loss + F.cross_entropy(logits, fut_ids[:, h].long())

                    # ---- SASRec InfoNCE aux loss ---- #
                    # Final decoder hidden state: [B, t5_d_model]
                    decoder_hidden = decoder_output_full[:, -1, :]
                    query = aux_head(decoder_hidden)  # [B, vae_embed_dim]

                    # next_item_ids: item indices for the target items
                    # tokenized_data.next_item_ids should contain corpus item indices
                    next_item_ids = tokenized_data.next_item_ids

                    aux_loss = sasrec_infonce_loss(
                        query=query,
                        positive_item_ids=next_item_ids,
                        item_embeddings=accelerator.unwrap_model(aux_head).item_embeddings,
                        n_negatives=sasrec_n_negatives,
                        temperature=sasrec_temperature,
                    )

                    loss = (sid_loss + current_lambda * aux_loss) / gradient_accumulate_every

                total_loss += loss.detach().item()
                total_sid_loss += (sid_loss.detach().item() / gradient_accumulate_every)
                total_aux_loss += (aux_loss.detach().item() / gradient_accumulate_every)

                if wandb_logging and accelerator.is_main_process:
                    train_debug_metrics = compute_debug_metrics(tokenized_data)

                accelerator.backward(loss)

            pbar.set_description(
                f"loss: {total_loss:.4f} | sid: {total_sid_loss:.4f} | aux: {total_aux_loss:.4f}"
            )

            accelerator.wait_for_everyone()

            # Gradient clipping is always applied in MTL (unlike vanilla train())
            accelerator.clip_grad_norm_(
                list(model.parameters()) + list(aux_head.parameters()),
                grad_clip_norm,
            )
            optimizer.step()
            lr_scheduler.step()

            accelerator.wait_for_everyone()

            if (iter + 1) % partial_eval_every == 0:
                model.eval()
                aux_head.eval()
                eval_loss = 0.0
                for batch in eval_dataloader:
                    data = batch_to(batch, device)
                    tokenized_data = tokenizer(data)
                    with torch.no_grad():
                        eval_loss = model(tokenized_data).loss.item()

                if wandb_logging and accelerator.is_main_process:
                    wandb.log({"eval_loss": eval_loss})

            if (iter + 1) % full_eval_every == 0:
                model.eval()
                aux_head.eval()
                with tqdm(
                    eval_dataloader,
                    desc=f"Eval {iter + 1}",
                    disable=not accelerator.is_main_process,
                ) as pbar_eval:
                    for batch in pbar_eval:
                        data = batch_to(batch, device)
                        tokenized_data = tokenizer(data)

                        with torch.no_grad():
                            generated = model.generate_next_sem_id(
                                tokenized_data, top_k=True, temperature=1
                            )

                        actual = tokenized_data.sem_ids_fut[:, :vae_n_layers]
                        metrics_accumulator.accumulate(
                            generated_ids=generated.sem_ids,
                            target_ids=actual,
                        )

                eval_metrics = metrics_accumulator.reduce()
                print(eval_metrics)
                if accelerator.is_main_process and wandb_logging:
                    wandb.log(eval_metrics)
                metrics_accumulator.reset()

            if accelerator.is_main_process:
                if (iter + 1) % save_model_every == 0 or iter + 1 == iterations:
                    state = {
                        "iter": iter,
                        "model": model.state_dict(),
                        "aux_head": accelerator.unwrap_model(aux_head).state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": lr_scheduler.state_dict(),
                        "lambda_aux_max": lambda_aux_max,
                        "sasrec_temperature": sasrec_temperature,
                    }

                    if not os.path.exists(save_dir_root):
                        os.makedirs(save_dir_root)

                    torch.save(state, save_dir_root + f"checkpoint_{iter}.pt")

                if wandb_logging:
                    wandb.log(
                        {
                            "learning_rate": optimizer.param_groups[0]["lr"],
                            "total_loss": total_loss,
                            "sid_loss": total_sid_loss,
                            "aux_infonce_loss": total_aux_loss,
                            "lambda_aux": current_lambda,
                            **train_debug_metrics,
                        }
                    )

            pbar.update(1)

    if wandb_logging:
        wandb.finish()


if __name__ == "__main__":
    parse_config()
    override_save_dir_for_sagemaker()
    train_mtl()
