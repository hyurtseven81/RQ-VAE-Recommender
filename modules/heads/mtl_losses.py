"""MTL loss functions and schedule for SASRec auxiliary training.

These are kept in a standalone module so they can be imported independently
from the training script (which has heavy dependencies on accelerate, wandb, etc.).
"""

import torch
import torch.nn.functional as F


def sasrec_infonce_loss(
    query: torch.Tensor,               # [B, d_item]
    positive_item_ids: torch.Tensor,   # [B]
    item_embeddings: torch.nn.Embedding,
    n_negatives: int = 1024,
    temperature: float = 0.05,
) -> torch.Tensor:
    """Sampled-softmax InfoNCE loss.

    Positives: actual next items.
    Negatives: in-batch (other items in batch) + uniform random sample.
    """
    B = query.shape[0]
    device = query.device

    # In-batch negatives: all items in the batch (includes the positive itself,
    # which is fine -- the diagonal entry is masked by the cross-entropy target)
    pos_embs = item_embeddings(positive_item_ids)  # [B, d_item]

    # Random negatives
    rand_ids = torch.randint(0, item_embeddings.num_embeddings, (n_negatives,), device=device)
    rand_embs = item_embeddings(rand_ids)  # [n_negatives, d_item]

    # Positive scores: [B]
    pos_scores = (query * pos_embs).sum(-1) / temperature

    # Negative scores: [B, B + n_negatives]
    # In-batch: query [B, d_item] @ pos_embs.T [d_item, B] -> [B, B]
    inbatch_scores = query @ pos_embs.T / temperature   # [B, B]
    rand_scores = query @ rand_embs.T / temperature     # [B, n_negatives]
    neg_scores = torch.cat([inbatch_scores, rand_scores], dim=1)  # [B, B + n_negatives]

    # InfoNCE: positive is index 0 in the concatenated logit vector
    # [B, 1 + B + n_negatives]
    all_scores = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)
    loss = F.cross_entropy(all_scores, torch.zeros(B, dtype=torch.long, device=device))
    return loss


class LambdaWarmupSchedule:
    """Linear warmup of lambda_aux from 0 to lambda_max over warmup_steps."""

    def __init__(self, lambda_max: float, warmup_steps: int):
        self.lambda_max = lambda_max
        self.warmup_steps = warmup_steps

    def get(self, step: int) -> float:
        if self.warmup_steps == 0:
            return self.lambda_max
        return self.lambda_max * min(1.0, step / self.warmup_steps)
