"""Eval-time invariants that catch silent train-vs-eval drift.

Two helpers exist because the §M-zero-metrics bug on `alpha-beauty-pilot`
(see `docs/runbook_sagemaker.md`) consumed two days of operator/agent time
producing a 27-row CSV of zeros before anyone realised the decoder ckpt
and the eval-time tokenizer were reading different SID tables.

* :func:`repair_eval_tokenizer` — overwrites ``tokenizer.cached_ids``
  with the train-time SID table loaded from the decoder ckpt's
  ``model.codebooks`` buffer. The decoder predicts SIDs in the
  train-time space, so eval-time tokenization MUST use that same
  table — anything else scores predictions against a mismatched
  key. With deterministic RQ-VAEs the recompute matches by accident;
  with partial-collapse codebooks (e.g. AGENTS.md fingerprints upstream
  beauty L0 at 48/256 codes) cuBLAS algorithm-selection non-determinism
  flips tied-code assignments and ~0.5% of rows drift. Always repair,
  not warn.

* :func:`assert_corpus_ids_match` — sanity check after repair, plus
  a hard error on the catastrophic case (different corpus size, which
  means dataset_split or processed cache changed).

* :func:`smoke_test_zero_alpha` — runs α=[0]*n_levels for a handful
  of batches BEFORE launching the full alpha grid. Per the AGENTS.md
  bypass invariant, α=[0]*n_levels MUST equal vanilla beam search;
  if recall is identically zero, the harness is broken — abort BEFORE
  burning compute on a uniformly-zero sweep.
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


def _recompute_dedup_column(codebooks: torch.Tensor) -> torch.Tensor:
    """Recompute the per-row dedup tag for a (N, n_levels) SID table.

    Mirrors the train-time logic in ``SemanticIdTokenizer.precompute_corpus_ids``:
    for each item ``i`` (in iteration order), the dedup tag is the count of
    earlier items ``j < i`` whose SID triple equals ``codebooks[i]``. The
    result keeps the (n_levels + 1)-tuple unique across all items, which is
    what the embedding-table indexing relies on.
    """
    n = codebooks.shape[0]
    dedup = torch.zeros(n, dtype=codebooks.dtype, device=codebooks.device)
    seen: dict[tuple, int] = {}
    rows = codebooks.tolist()
    for i, row in enumerate(rows):
        key = tuple(row)
        dedup[i] = seen.get(key, 0)
        seen[key] = seen.get(key, 0) + 1
    return dedup


def repair_eval_tokenizer(
    model, tokenizer: SemanticIdTokenizer, n_levels: int
) -> None:
    """Force the tokenizer's cached_ids to match the decoder's saved table.

    Must be called AFTER ``model.load_state_dict(...)`` (so ``model.codebooks``
    holds the train-time buffer). Replaces the first ``n_levels`` columns of
    ``tokenizer.cached_ids`` with ``model.codebooks`` and recomputes the
    dedup column for uniqueness. Reports row-drift count if any.
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
            "before repair."
        )
    eval_cached_cpu = eval_cached.detach().cpu().long()

    if train_codebooks.shape != eval_cached_cpu[:, :n_levels].shape:
        raise RuntimeError(
            f"corpus_ids shape mismatch: train-time={tuple(train_codebooks.shape)} "
            f"eval-time={tuple(eval_cached_cpu[:, :n_levels].shape)}. The corpus has "
            f"changed between training and eval — different dataset_split or "
            f"processed cache. Refuse to evaluate."
        )

    n = train_codebooks.shape[0]
    eval_codebooks = eval_cached_cpu[:, :n_levels]
    n_mismatch = int((train_codebooks != eval_codebooks).any(dim=1).sum().item())
    if n_mismatch > 0:
        per_level = (train_codebooks != eval_codebooks).sum(dim=0).tolist()
        print(
            f"[repair] corpus_ids drift: {n_mismatch}/{n} rows differ "
            f"({100.0 * n_mismatch / n:.2f}%). Per-level: " +
            "  ".join(f"lvl{i}={c}" for i, c in enumerate(per_level)) +
            ". Replacing eval-time SIDs with train-time codebooks for the "
            "decoder's saved table."
        )
    else:
        print(f"[repair] corpus_ids already match ({n}/{n} rows); repair is a no-op.")

    dedup = _recompute_dedup_column(train_codebooks)
    repaired = torch.cat([train_codebooks, dedup.unsqueeze(1)], dim=1)
    tokenizer.cached_ids = repaired.to(eval_cached.device)


def assert_corpus_ids_match(model, tokenizer: SemanticIdTokenizer, n_levels: int) -> None:
    """Tautological check after :func:`repair_eval_tokenizer` — keep as a
    belt-and-suspenders sanity guard.
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
            "Tokenizer has no `cached_ids` — repair was skipped or failed."
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
            f"corpus_ids drift after repair: {n_mismatch}/{n} rows still differ. "
            f"Per-level: " +
            "  ".join(f"lvl{i}={c}" for i, c in enumerate(per_level)) +
            ". `repair_eval_tokenizer` failed to sync — investigate."
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
