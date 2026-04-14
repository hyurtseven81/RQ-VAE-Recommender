import argparse

import gin
import torch

from data.schemas import TokenizedSeqBatch


def eval_mode(fn):
    def inner(self, *args, **kwargs):
        was_training = self.training
        self.eval()
        out = fn(self, *args, **kwargs)
        self.train(was_training)
        return out

    return inner


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", nargs="?", type=str, default=None,
                        help="Path to gin config file.")
    parser.add_argument("--config_path", dest="config_path_flag", type=str, default=None)
    args, _ = parser.parse_known_args()
    path = args.config_path or args.config_path_flag
    if path is None:
        parser.error("config_path is required (positional or --config_path)")
    gin.parse_config_file(path)


def override_save_dir_for_sagemaker():
    """If running on SageMaker, override save_dir_root to write to /opt/ml/model/."""
    import os
    sm_model_dir = os.environ.get("SM_MODEL_DIR")
    if sm_model_dir is None:
        return
    # Redirect output
    for fn_name in ("train.save_dir_root", "train_mtl.save_dir_root"):
        try:
            gin.bind_parameter(fn_name, sm_model_dir + "/")
        except ValueError:
            pass
    # Disable wandb if no API key is configured
    if not os.environ.get("WANDB_API_KEY"):
        os.environ["WANDB_MODE"] = "disabled"


@torch.no_grad
def compute_debug_metrics(
    batch: TokenizedSeqBatch, model_output=None, prefix: str = ""
) -> dict:
    seq_lengths = batch.seq_mask.sum(axis=1).to(torch.float32)
    prefix = prefix + "_"
    debug_metrics = {
        prefix + f"seq_length_p{q}": torch.quantile(seq_lengths, q=q)
        .detach()
        .cpu()
        .item()
        for q in [0.25, 0.5, 0.75, 0.9, 1]
    }
    if model_output is not None:
        loss_debug_metrics = {
            prefix + f"loss_{d}": model_output.loss_d[d].detach().cpu().item()
            for d in range(batch.sem_ids_fut.shape[1])
        }
        debug_metrics.update(loss_debug_metrics)
    return debug_metrics
