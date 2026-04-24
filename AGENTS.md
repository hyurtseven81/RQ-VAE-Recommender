# AGENTS.md — RQ-VAE Level-Aware Hybrid Decoding

Key facts for AI assistants working in this repo. See `docs/paper_plan.md`
for the current two-stage experimental plan (CIKM 2026). Collapse-investigation
material lives in `docs/bisect_runbook.md` and is **parked** — it is
orthogonal to the paper contribution.

## Paper thesis (scope)

The contribution is **decoding strategies over RQ-VAE semantic IDs**, not the
tokeniser. Upstream EdoardoBotta pre-trained RQ-VAE checkpoints are the
fixed tokeniser layer; codebook collapse is a property of those checkpoints
and is **not** something this paper claims to fix.

- **Stage 1** — Recall@{5,10,20} + NDCG@{5,10,20} across α-free beam-search
  strategies on a vanilla SID-only decoder (`train_decoder.py`).
  Strategies: `vanilla`, `dbs`, `gumbel_topk`, `hybrid`, `sasrec_rerank`.
- **Stage 2** — Same metrics with an MTL decoder (`train_decoder_mtl.py`,
  SASRec auxiliary head) plus level-aware α mixing. Strategies:
  `level_aware_mix{,_grid,_learned}`. Pilot 3³ → refined 5³ α grid, plus
  learned-α variant.

Datasets (all three): Amazon Beauty, Amazon Sports, MovieLens-32M — all
from upstream `trained_models/` checkpoints.

## Architecture constants

- **Codebook levels**: 3 (`n_hierarchies = 3`)
- **Codes per level**: 256 (`num_embeddings_per_hierarchy = 256`)
- **Codebook dim**: 32 (`embed_dim = 32`) — also `d_item` in SASRecAuxHead
- **Decoder hidden dim**: 384 (`d_model = 384`)
- **Beam search is sampling-based** (multinomial), NOT greedy. `generate()` uses `torch.multinomial`. `n_cands` raised to `min(200, vocab_size)`.

## Critical invariants

- **alpha=0 bypass**: In `LevelAwareHybridDecoding`, when `alpha[h] == 0.0` the original beam probas are passed through unchanged — this guarantees exact base strategy output. Do not break this.
- **Z-score normalization**: both `log_p` and `s_dense` are z-scored across vocab before mixing.
- **Partial reconstruction**: `r_h^cand = Σ_{i<h} e_{c_i} + e_{c_h}^cand` — used for dense scoring at level h, not the full reconstruction.
- **LIGER baselines are PLACEHOLDERS**: `results/baselines/liger_paper_numbers.json` contains placeholder values. Verify against actual LIGER paper Table 1 before submission.

## Training

- `train_decoder.py` — SID-only cross-entropy, no grad clipping. Stage 1.
- `train_decoder_mtl.py` — joint `L_sid + λ(t)·L_sasrec`; **grad clip norm 1.0** (deviation from vanilla); λ warmup 0→0.2 over first 10% of steps. Stage 2.
- SASRecAuxHead item embeddings are **initialized from RQ-VAE encoder outputs** (collaborative signal, 32-d), not from Sentence-T5 text embeddings like LIGER.
- Item features are truncated to `vae_input_dim` before passing to encoder (ML32M/ML1M have 786-dim features but RQ-VAE uses 768).

## Module layout

```
modules/decoding/      — beam search strategies (vanilla, dbs, gumbel, hybrid, level_aware_mix, sasrec_reranker)
modules/heads/         — sasrec_head.py, mtl_losses.py
modules/analysis/      — residual_entropy.py
evaluate/              — metrics.py, stats.py, result_store.py, run_eval.py, validate_rqvae.py
scripts/               — alpha_grid_search.py, train_alpha_params.py, make_figures.py, make_tables.py
sagemaker/             — launch scripts for SageMaker training + eval jobs
paper/                 — LaTeX source (CIKM 2026 submission)
docs/                  — paper_plan.md, bisect_runbook.md (parked), implementation_plan.md
```

## AWS / S3

Account-specific identifiers are read from environment variables so they
never land in the repository. The launch scripts read them from
`sagemaker/launch/_aws_env.py`; other scripts use `os.environ.get(...)`
directly. Required on any machine that launches jobs:

```
export RQVAE_S3_BASE=s3://<your-bucket>/rqvae-level-aware   # required
export RQVAE_SAGEMAKER_ROLE=arn:aws:iam::<acct-id>:role/<role-name>   # required
export RQVAE_AWS_PROFILE=<boto3-profile-name>    # optional; falls back to AWS_PROFILE
export RQVAE_AWS_REGION=us-east-1                # optional; falls back to AWS_REGION, then us-east-1
```

- **Instance quotas**: `g5.xlarge` on-demand=1 (shared), `g5.4xlarge` on-demand=30 (use this for RQ-VAE / validator), `g5.xlarge` spot=5 (fine for eval sweeps and α pilot).
- **Auth refresh**: whatever your org requires (e.g. `mwinit`, `aws sso login --profile $RQVAE_AWS_PROFILE`) — credentials expire roughly every 10h.

## Datasets and upstream checkpoints

Paper uses three datasets, all with upstream-committed RQ-VAE checkpoints
from `EdoardoBotta/RQ-VAE-Recommender` under `trained_models/`. We do not
retrain the RQ-VAE layer for the paper; we train only the decoder on top.

| Dataset | Users | Items | Upstream RQ-VAE checkpoint | Codebook fingerprint (end-of-training validator output) |
|---|---:|---:|---|---|
| Amazon Beauty | ~22K | ~12K | `trained_models/rqvae_amazon_beauty/checkpoint_*.pt` | Healthy — 248–256/256 per level, entropy 7.66–7.73 bits (our 2026-04-21 validation) |
| Amazon Sports | ~35K | ~18K | `trained_models/rqvae_amazon_sports/checkpoint_high_entropy.pt` | Partial collapse — L0: 33/256 (1.78 bits), L1: 96/256 (3.81 bits), L2: 134/256 (4.82 bits). Accepted as the "imperfect-codebook" reference. |
| MovieLens-32M | ~200K | ~86K | `trained_models/rqvae_ml32m/checkpoint_high_entropy.pt` | **Unvalidated** — Stage 0 of the paper plan gates inclusion pending (a) validator fingerprint and (b) fork data pipeline compatibility check. |

Upstream HEAD's `trained_models/` is **not** tracked in this fork
(`trained_models/` is in `.gitignore` since commit `6355423`). To fetch
them locally:

```bash
git remote add upstream https://github.com/EdoardoBotta/RQ-VAE-Recommender.git
git fetch upstream main
git checkout upstream/main -- trained_models/
# (then ignore / stash; do not commit)
```

### ML32M pipeline compatibility — known risks

ML1M was previously dropped for five pipeline issues (AGENTS.md history).
One is fixed (feature-dim truncation, commit `55abeba`). Four may still
apply to ML32M and need a smoke check before decoder training:

- Missing `is_train` / `text` fields in item data
- Split-name mismatch (`eval` vs `test`)
- `max_seq_len` padding (upstream used 200, fork uses 20)
- CUDA index-out-of-bounds from padding tokens

Gate: `docs/paper_plan.md` → Stage 0. Not blocked on resolving these
today — just on deciding include/drop for ML32M.

## Data prep

- **Amazon (Beauty, Sports)**: auto-download via `AmazonReviews.download()` (single Google-Drive zip covers both splits plus Toys if ever needed). Preprocessed features cached at `$RQVAE_S3_BASE/datasets/amazon/` for Beauty + Sports.
- **MovieLens-32M**: auto-download via `RawMovieLens32M` in `data/ml32m.py`. Caching policy TBD — part of Stage 0.
- Training jobs that mount `s3://.../datasets/<name>/` as a `dataset` channel skip preprocessing — `override_save_dir_for_sagemaker()` symlinks `SM_CHANNEL_DATASET` onto the gin-configured `dataset_folder`.
- All datasets use leave-one-out split.
- Sentence-T5 features are already L2-normalized (verified Beauty + Sports: norm 0.9995–1.0005).

## Pipeline stages

```
docs/paper_plan.md Stage 0      — ckpt inventory + ML32M pipeline gate
  ↓
Stage 1 vanilla decoder × 3     — train_decoder.py per dataset
  + α-free eval sweep           — evaluate/run_eval.py × 5 strategies × 3 datasets
  ↓
Stage 2 MTL decoder × 3         — train_decoder_mtl.py per dataset
  + α pilot grid 3³             — evaluate/alpha_search.py (pilot stage)
  + α refined grid 5³           — evaluate/alpha_search.py (refine stage)
  + learned α                   — evaluate/alpha_train.py
  ↓
scripts/collect_results.py      — aggregate parquets
scripts/make_tables.py          — paper tables
scripts/make_figures.py         — α heatmaps
```

Stage 1 and Stage 2 can run in parallel once Stage 0 confirms checkpoints
are loadable.

## Known issues and workarounds

### SageMaker dependency pins (requirements.txt)
Critical — relaxing them breaks SageMaker containers:
- `polars==1.9.0` — newer versions change `.list.to_array()` return type from List to Array, breaking `_df_to_tensor_dict`.
- `wandb>=0.19.0,<0.25` — wandb 0.25+ requires protobuf>=6.32.1 but SM container has 6.31.1; crashes at import time before `WANDB_MODE=disabled` takes effect.
- `gin_config==0.5.0` — exact pin required.
- Do NOT pin `triton` — SM container ships the correct version for its torch build; overriding it breaks `torch._inductor`.

### SageMaker runtime
- `override_save_dir_for_sagemaker()` in `modules/utils.py` redirects output to `/opt/ml/model/` and disables wandb when no API key is present.
- `parse_config()` accepts both positional args (local) and `--config_path` flag (SageMaker).
- `source_dir="."` uploads the entire repo — keep `trained_models/` git-ignored but present locally so it's included in the SageMaker upload.
- Spot instances on g5.xlarge/g5.2xlarge are unreliable (frequent interruptions, no checkpointing); use **g5.4xlarge on-demand** for RQ-VAE / validator jobs. Spot is fine for short eval jobs where an interruption just retries.

### Load-time RQ-VAE handling
Downstream scripts that load an RQ-VAE checkpoint manually must set
`layer.do_kmeans_init = False` on each `Quantize` layer immediately after
`load_state_dict`. `evaluate/validate_rqvae.py` does this. Most other
callers route through `SemanticIdTokenizer` which passes
`codebook_kmeans_init=False`. If you add a new loader, do the same.

### ML1M dataset
Dropped for this paper — five pipeline incompatibilities (see "ML32M
pipeline compatibility" above). MovieLens-32M is the substitute.

## Parked: codebook-collapse investigation

The fork attempted four rounds of RQ-VAE retraining (`repro`, `repro2`,
`repro3`, `sports_v{6,7}`) and all validated as collapsed despite two
reverts (`f9b645a` for the L2-norm "fix", `4e0eb00` + `baa1190` for the
`kmeans_initted` buffer). The precise cause remains undiagnosed.

This does **not** affect the paper. The paper uses upstream pre-trained
RQ-VAE checkpoints — fork retraining is not on the critical path.
See `docs/bisect_runbook.md` for the experiments that would diagnose the
collapse if ever revisited. The bisect levers
(`train.vae_mlp_activation`, `train.vae_decoder_normalize`,
`RQVAE_DISABLE_COMPILE` env var, `configs/rqvae_amazon_beauty_bisect_*.gin`)
remain wired in but are unused.

## Test suite

```bash
pytest tests/ -q
pytest tests/ --cov --cov-fail-under=80
```

Known gaps: `tests/integration/` is empty, no `tests/data/test_steam_loader.py`,
no `tests/data/test_ml32m_loader.py`.

## Decoding strategy registry

Strategies registered in `modules/decoding/__init__.py`:

- **Stage 1 (α-free)**: `vanilla`, `dbs`, `gumbel_topk`, `hybrid`, `sasrec_rerank`
- **Stage 2 (level-aware α)**: `level_aware_mix`, `level_aware_mix_grid`, `level_aware_mix_learned`

Post-hoc vs in-loop: `sasrec_rerank` is post-hoc (reranks completed beams
using a single per-user query); the three `level_aware_mix*` variants mix
dense + autoregressive scores in-loop at every decoding level. The
`run_eval.py` harness handles both — for `sasrec_rerank` it runs a
`VanillaBeamSearch` pass with `return_query_hidden=True` and invokes
`SASRecReranker.rerank` after generation.

## Entry points for Stage 1 / Stage 2

| Role | Script | Launcher |
|---|---|---|
| Decoder training (α-free) | `train_decoder.py` | `sagemaker/launch/launch_decoder.py` |
| Decoder training (MTL, SASRec aux head) | `train_decoder_mtl.py` | `sagemaker/launch/launch_mtl.py` |
| Single strategy eval | `evaluate/run_eval.py` | `sagemaker/launch/launch_decoding_eval.py` |
| Per-level α grid sweep (pilot + refined) | `evaluate/alpha_search.py` | `sagemaker/launch/launch_alpha_search.py` |
| Learned-α training | `evaluate/alpha_train.py` | `sagemaker/launch/launch_alpha_params.py` |
| RQ-VAE ckpt validation | `evaluate/validate_rqvae.py` | `sagemaker/launch/launch_validate_rqvae.py` |
| Local pre-flight gate | `scripts/stage0_pipeline_check.py` | — |

## Keeping this file up to date

Update AGENTS.md whenever:
- A Stage-0/1/2 milestone completes — update dataset fingerprints, add decoder ckpt URIs to the datasets table.
- A new dataset is added or dropped (Stage 0 may drop ML32M; propagate here).
- A dependency pin changes in `requirements.txt` (add to known issues if SageMaker-related).
- A new decoding strategy is registered in `modules/decoding/__init__.py`.
- A new workaround or known issue is discovered.

The datasets table and pipeline stages section are the most frequently
stale — check them first.
