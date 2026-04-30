"""Eval-time invariants that catch silent train-vs-eval drift.

Both helpers exist because the §M-zero-metrics bug on `alpha-beauty-pilot`
(see `docs/runbook_sagemaker.md`) consumed two days of operator/agent time
producing a 27-row CSV of zeros before anyone realised the decoder ckpt
and the eval-time tokenizer were reading different SID tables. These
invariants are the cheapest possible defence:

* :func:`assert_corpus_ids_match` — compares the decoder ckpt's saved
  ``codebooks`` buffer against the freshly recomputed
  ``tokenizer.cached_ids`` and aborts on any row mismatch. Catches any
  drift in pretrained_rqvae_path / dataset_split / processed-cache /
  RQ-VAE-arch between training and eval.
* :func:`smoke_test_zero_alpha` — runs α=[0]*n_levels for a handful of
  batches BEFORE launching the full alpha grid. Per the AGENTS.md
  bypass invariant, α=[0]*n_levels MUST equal vanilla beam search;
  if recall is identically zero, the harness is broken (or, if (1)
  passed, something subtler is — e.g. accumulator regression or
  bypass code path).

Both raise ``RuntimeError`` on failure with actionable messages.
"""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from data.utils import batch_to
from evaluate.metrics import TopKAccumulator
from modules.decoding.level_aware_mix import LevelAwareHybridDecoding
from modules.decoding.vanilla import VanillaBeamSearch
from modules.heads.sasrec_head import SASRecAuxHead
from modules.tokenizer.semids import SemanticIdTokenizer


def assert_corpus_ids_match(model, tokenizer: SemanticIdTokenizer, n_levels: int) -> None:
    """Hard-error if the decoder's saved codebooks disagree with eval-time SIDs.

    Must be called AFTER ``model.load_state_dict(...)`` so that
    ``model.codebooks`` holds the train-time buffer (set in
    ``train_decoder.py`` from ``tokenizer.cached_ids[:, :n_levels]`` at
    training time). The eval-time tokenizer's ``cached_ids`` is the
    freshly recomputed table — if anything has drifted, every row
    will differ and recall will be identically zero downstream.
    """
    train_codebooks = getattr(model, "codebooks", None)
    if train_codebooks is None:
        raise RuntimeError(
            "Decoder model has no `codebooks` buffer post-load — either the "
            "checkpoint is from an older format or `load_state_dict(strict=False)` "
            "silently dropped it. Refuse to evaluate."
        )
    train_codebooks = train_codebooks.detach().cpu().long()
    eval_cached = tokenizer.cached_ids
    if eval_cached is None:
        raise RuntimeError(
            "Tokenizer has no `cached_ids` — `precompute_corpus_ids` was not run "
            "before invariant check."
        )
    eval_codebooks = eval_cached[:, :n_levels].detach().cpu().long()

    if train_codebooks.shape != eval_codebooks.shape:
        raise RuntimeError(
            f"corpus_ids shape mismatch: train-time={tuple(train_codebooks.shape)} "
            f"eval-time={tuple(eval_codebooks.shape)}. The corpus has changed "
            f"between training and eval — different dataset_split or processed cache."
        )

    n = train_codebooks.shape[0]
    n_mismatch = int((train_codebooks != eval_codebooks).any(dim=1).sum().item())
    if n_mismatch > 0:
        per_level = (train_codebooks != eval_codebooks).sum(dim=0).tolist()
        raise RuntimeError(
            f"corpus_ids drift: {n_mismatch}/{n} rows differ between train-time "
            f"(saved in ckpt) and eval-time (recomputed by tokenizer). "
            f"Per-level disagreement: " +
            "  ".join(f"lvl{i}={c}" for i, c in enumerate(per_level)) +
            ". Likely cause: the gin's `pretrained_rqvae_path` and the launcher's "
            "`--rqvae-ckpt` resolve to different files. Run "
            "`scripts/diag_corpus_ids.py` for a deeper diff. Refuse to evaluate."
        )
    print(f"[invariant] corpus_ids match: {n}/{n} rows agree across {n_levels} levels.")


def smoke_test_zero_alpha(
    model,
    tokenizer: SemanticIdTokenizer,
    eval_dataloader: DataLoader,
    aux_head: SASRecAuxHead,
    codebook_embs: list[torch.Tensor],
    device: torch.device,
    n_levels: int,
    n_batches: int = 5,
) -> None:
    """Run α=[0]*n_levels on a few batches and abort if all metrics are zero.

    Per ``modules/decoding/level_aware_mix.py`` and the AGENTS.md bypass
    invariant, ``α=[0,...,0]`` must be byte-equivalent to the base
    strategy (here ``VanillaBeamSearch``). On a working setup this gives
    non-zero recall; if it gives zero, every alpha point in the grid
    will too, and we should fail fast instead of burning the full sweep.
    """
    strategy = LevelAwareHybridDecoding(
        base_strategy=VanillaBeamSearch(),
        alpha=[0.0] * n_levels,
        aux_head=aux_head,
    )
    acc = TopKAccumulator(ks=[5, 10, 20])
    n_seen = 0
    for i, batch in enumerate(eval_dataloader):
        if i >= n_batches:
            break
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
        n_seen += target.shape[0]

    metrics = acc.reduce()
    aggregate = {k: float(v) for k, v in metrics.items() if k != "per_user"}
    if all(v == 0.0 for v in aggregate.values()):
        raise RuntimeError(
            f"α=[0]*{n_levels} smoke test produced zero recall/ndcg across "
            f"{n_seen} held-out targets. Per the bypass invariant this means the "
            f"alpha grid will be uniformly zero — abort BEFORE wasting compute. "
            f"Likely causes: alpha=0 bypass broken in "
            f"`LevelAwareHybridDecoding.expand`; accumulator regression in "
            f"`evaluate/metrics.py`; decoder ckpt not loaded properly (check the "
            f"`assert_corpus_ids_match` invariant didn't fire above). "
            f"Aggregates seen: {aggregate}"
        )
    print(f"[invariant] α=[0]*{n_levels} smoke test PASS over {n_seen} targets: " +
          "  ".join(f"{k}={v:.4f}" for k, v in aggregate.items()))
