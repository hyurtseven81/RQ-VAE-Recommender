# Paper plan — RQ-VAE Level-Aware Hybrid Decoding (CIKM 2026)

## Thesis

Decoding strategies over RQ-VAE semantic IDs materially affect sequential
recommendation quality. Concretely:

- **H1**: Beam-search variants (diversity penalty, temperature sampling, hybrid
  dense-reranking) change Recall/NDCG vs. plain beam search on SID sequences.
- **H2**: Adding a recall-focused SASRec auxiliary head to the decoder and
  mixing its scores into beam search with per-level α weights further improves
  Recall/NDCG; the optimal α schedule varies per codebook level.

Codebook quality is a property of the tokenizer and is treated as given. We
do **not** claim codebook improvements; fixing RQ-VAE collapse is orthogonal
(see `docs/bisect_runbook.md`).

## Datasets

All three use upstream EdoardoBotta/RQ-VAE-Recommender pre-trained RQ-VAE
checkpoints committed under `trained_models/` on upstream `main`. Those
checkpoints are not re-trained as part of the paper.

| Dataset | Upstream RQ-VAE checkpoint | Notes |
|---|---|---|
| Amazon Beauty | `trained_models/rqvae_amazon_beauty/checkpoint_*.pt` | Healthy codebook per our 2026-04-21 validation (248–256/256, 7.7 bits) |
| Amazon Sports | `trained_models/rqvae_amazon_sports/checkpoint_high_entropy.pt` | Partial-collapse codebook per 2026-04-22 validation (L0: 33 codes, 1.78 bits; L1: 96; L2: 134) — accepted as the "imperfect codebook" reference point |
| MovieLens-32M | `trained_models/rqvae_ml32m/checkpoint_high_entropy.pt` | Pipeline compatibility unverified — Stage 0 below gates inclusion |

Two-dataset fallback if ML32M Stage 0 fails: Beauty + Sports with codebook-quality
contrast as the generalisation story.

## Metrics

`Recall@{5,10,20}` and `NDCG@{5,10,20}` on the leave-one-out held-out target.
Reported as mean over users with a paired-bootstrap CI for the learned-α
vs. baseline comparison (see `evaluate/stats.py`).

## Stage 0 — inventory and pipeline validation

1. Confirm all three upstream checkpoints are available locally
   (`ls trained_models/rqvae_{amazon_beauty,amazon_sports,ml32m}/`). If any
   is missing, fetch from upstream repo.
2. Run `evaluate/validate_rqvae.py` against each checkpoint and record the
   per-level `unique_sids / entropy_bits / min_codebook_distance` as the
   "codebook fingerprint" in the final paper's setup table.
3. **ML32M pipeline gate**: run a smoke forward pass through the fork's
   current data loader on a small ML32M batch. Failure modes to check:
   feature-dim mismatch, missing `is_train` / `text` fields, split naming
   (`eval` vs `test`), `max_seq_len` padding, CUDA index-out-of-bounds.
   Known-fixed: feature-dim truncation (commit `55abeba`). If any other
   failure appears, the Stage-0 decision is: (a) fix it within one working
   day, (b) swap in MovieLens-1M if upstream has a suitable ckpt, or
   (c) drop to Beauty + Sports.

Gating output: `docs/paper_plan_stage0_report.md` with the three fingerprints
and the ML32M-include decision.

## Stage 1 — α-free beam-search strategies

Per dataset, train one **vanilla** decoder (`train_decoder.py`, SID-only
cross-entropy, no auxiliary head) against the upstream RQ-VAE, then run the
held-out eval sweep.

### Training

- Entry: `train_decoder.py` with the corresponding `configs/decoder_*.gin`.
- `pretrained_rqvae_path` points to the upstream RQ-VAE checkpoint for the
  dataset.
- Launch: `sagemaker/launch/launch_rqvae.py`'s sibling pattern — needs
  analogous `sagemaker/launch/launch_decoder.py` that accepts
  `--pretrained-rqvae` as an override. Tracked as an implementation todo
  (one of the few code changes required for this plan).

### Strategies evaluated

Only the α-free strategies from `modules/decoding/__init__.py`:

- `vanilla` — standard beam search, the baseline.
- `dbs` — diverse beam search with per-group Hamming diversity penalty.
- `gumbel_topk` — temperature-scaled Gumbel-top-k sampling.
- `hybrid` — per-step mix of log p(SID) and dense reconstruction distance
  with a scalar α (kept in Stage 1 because α is a **single scalar**, not a
  per-level schedule — the paper distinguishes these as "hybrid" vs
  "level-aware-hybrid").
- `sasrec_rerank` — item-level rerank of the top-B beams by SASRec dense
  scores. This is α-free at the rerank stage but requires an already-trained
  SASRec head; for Stage 1 we reuse the head from the Stage-2 MTL decoder as
  a rerank-only oracle and flag this in the paper's method section.

Beam size = 10 (standard in TIGER). Sampling temperature / diversity penalty
grids kept minimal (3 points each) so Stage-1 stays a comparison, not a
hyper-parameter study.

### Outputs

- `results/stage1/{dataset}/{strategy}.json` per-run.
- `results/stage1/all_runs.parquet` aggregated.
- Table in `paper/`: "Stage 1 — Decoding strategy comparison (no α)" with
  rows = (dataset × strategy), columns = Recall/NDCG @ {5,10,20}.

## Stage 2 — SASRec head + per-level α mixing

Per dataset, train one **MTL** decoder (`train_decoder_mtl.py`, joint
`L_sid + λ(t)·L_sasrec`) against the upstream RQ-VAE. Then evaluate the
α-parameterised decoding strategies.

### Training

- Entry: `train_decoder_mtl.py` with the corresponding
  `configs/decoder_{amazon_beauty,amazon_sports,ml32m}_mtl.gin` (the last
  may need to be authored; current set is Amazon-only).
- `pretrained_rqvae_path` same as Stage 1.
- Grad-clip 1.0, λ warm-up 0→0.2 over first 10% of steps (AGENTS.md).

### Strategies evaluated

The α-parameterised family from `modules/decoding/__init__.py`:

- `level_aware_mix` — user-provided fixed per-level α = (α₀, α₁, α₂).
- `level_aware_mix_grid` — exhaustive per-level α grid search
  (`scripts/alpha_grid_search.py` runs this as a sweep).
- `level_aware_mix_learned` — gradient-learned per-level α on a validation
  split (`scripts/train_alpha_params.py`).

### α grid strategy

Two-pass grid. Prevents burning compute on a dense grid over a region that
turns out to be flat.

1. **Pilot** (27 points × 3 datasets = 81 runs, a few hours each on
   g5.xlarge): α ∈ {0, 0.5, 1.0} independently per level, i.e. the
   3×3×3 cube. Report best point per dataset.
2. **Refine** (125 points × 3 datasets = 375 runs): 5×5×5 grid centred on
   each dataset's pilot winner, step 0.1.
3. **Learned-α** baseline: single run per dataset with
   `level_aware_mix_learned`, initialised at the pilot winner. Reports the
   learned α schedule, useful as "oracle" upper bound.

The refined-grid best point is the headline number; the learned α is the
"no-tuning" story.

### Outputs

- `results/stage2/{dataset}/{alpha_schedule_or_variant}.json` per-run.
- `results/stage2/all_runs.parquet`, `results/stage2/per_user_runs.parquet`
  (latter for paired-bootstrap CI).
- Figure: per-dataset heatmap of NDCG@10 across (α₀, α₁) with α₂ pinned at
  the refined winner, to visualise which levels benefit from dense mixing.
- Table: "Stage 2 — Level-aware α results" with best fixed α schedule,
  learned α schedule, and deltas vs. Stage 1 `vanilla` baseline.

## Checkpoint dependency graph

```
upstream RQ-VAE (per dataset)
  ├── Stage 1 vanilla decoder (per dataset)
  │     └── eval sweep × 5 α-free strategies
  └── Stage 2 MTL decoder (per dataset)
        ├── pilot α grid 3×3×3
        ├── refined α grid 5×5×5 (centred on pilot winner)
        └── learned-α run
```

Stage 2 does **not** depend on Stage 1 decoder checkpoints — they're
trained independently. Stage 1 and Stage 2 can run in parallel once
Stage 0 confirms checkpoints are loadable.

## Time / compute budget

Per-dataset, spot-friendly where noted:

| Step | Instance | Wall-clock | Parallelism |
|---|---|---|---|
| Stage 0 validator run | g5.4xlarge on-demand | ~20 min × 3 | serial (cheap) |
| Stage 1 decoder training | g5.2xlarge spot | ~12–20 h × 3 | parallel |
| Stage 1 eval sweep | g5.xlarge spot | ~1 h × 15 runs | parallel |
| Stage 2 MTL decoder training | g5.2xlarge spot | ~15–24 h × 3 | parallel |
| Stage 2 α pilot | g5.xlarge spot | ~1 h × 81 | ~20-wide parallel |
| Stage 2 α refine | g5.xlarge spot | ~1 h × 375 | ~20-wide parallel |
| Stage 2 learned-α | g5.2xlarge on-demand | ~4 h × 3 | parallel |

Estimated wall-clock to paper-ready tables (assuming no retries):
Stage 0 half-day, Stage 1 two days, Stage 2 four days. Five working days
end-to-end if nothing collapses.

## What the plan intentionally does not include

- No RQ-VAE retraining. Upstream checkpoints are the tokeniser layer.
- No codebook-collapse investigation beyond the validator's one-page
  fingerprint per dataset. The bisect experiments (see
  `docs/bisect_runbook.md`) remain parked.
- No Toys / Steam. Upstream has no pre-trained RQ-VAE for those datasets in
  the fork's lineage; re-training would cost the same collapse-investigation
  detour the paper is scoped to avoid.

## Implementation todos unlocked by this plan

Tracked separately from the experimental work — these are one-time code
changes the plan depends on:

1. `sagemaker/launch/launch_decoder.py` (new) — parallels `launch_mtl.py`
   but runs `train_decoder.py` (vanilla). Takes `--pretrained-rqvae`.
2. `configs/decoder_ml32m.gin` and `configs/decoder_ml32m_mtl.gin` — may
   already exist; otherwise author from the `decoder_amazon_beauty*.gin`
   template.
3. Stricter `evaluate/validate_rqvae.py` "healthy" criterion (informational
   only — not a gate) so the paper's setup table can report codebook
   fingerprints consistently.
4. `scripts/make_tables.py` / `scripts/make_figures.py` already exist per
   `modules/` layout; update to consume the Stage 1 / Stage 2 parquets
   when they land.
