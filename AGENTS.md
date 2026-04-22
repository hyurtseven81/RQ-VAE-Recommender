# AGENTS.md — RQ-VAE Level-Aware Hybrid Decoding

Key facts for AI assistants working in this repo. See `docs/implementation_plan.md` for full design decisions.

## Architecture constants

- **Codebook levels**: 3 (`n_hierarchies = 3`)
- **Codes per level**: 256 (`num_embeddings_per_hierarchy = 256`)
- **Codebook dim**: 32 (`embed_dim = 32`) — this is also `d_item` in SASRecAuxHead
- **Decoder hidden dim**: 384 (`d_model = 384`)
- **Beam search is sampling-based** (multinomial), NOT greedy. `generate()` uses `torch.multinomial`. n_cands raised to `min(200, vocab_size)`.

## Critical invariants

- **alpha=0 bypass**: In `LevelAwareHybridDecoding`, when `alpha[h] == 0.0` the original beam probas are passed through unchanged — this guarantees exact base strategy output. Do not break this.
- **Z-score normalization**: both `log_p` and `s_dense` are z-scored across vocab before mixing.
- **Partial reconstruction**: `r_h^cand = Σ_{i<h} e_{c_i} + e_{c_h}^cand` — used for dense scoring at level h, not the full reconstruction.
- **LIGER baselines are PLACEHOLDERS**: `results/baselines/liger_paper_numbers.json` contains placeholder values. Verify against actual LIGER paper Table 1 before submission.

## Training

- `train_decoder.py` — standard SID cross-entropy, no grad clipping
- `train_decoder_mtl.py` — joint `L_sid + λ(t)·L_sasrec`; **grad clip norm 1.0** (deviation from vanilla); λ warmup 0→0.2 over first 10% of steps
- SASRecAuxHead item embeddings are **initialized from RQ-VAE encoder outputs** (collaborative signal, 32-d), not from Sentence-T5 text embeddings like LIGER
- Item features are truncated to `vae_input_dim` before passing to encoder (ML1M has 786-dim features but RQ-VAE uses 768)

## Module layout

```
modules/decoding/      — beam search strategies (vanilla, dbs, gumbel, hybrid, level_aware_mix, sasrec_reranker)
modules/heads/         — sasrec_head.py, mtl_losses.py
modules/analysis/      — residual_entropy.py
evaluate/              — metrics.py, stats.py, result_store.py, run_eval.py
scripts/               — alpha_grid_search.py, train_alpha_params.py, make_figures.py, make_tables.py
sagemaker/             — launch scripts for SageMaker training jobs
paper/                 — LaTeX source (CIKM 2026 submission)
```

## AWS / S3

- **Bucket**: `s3://REDACTED-BUCKET/rqvae-level-aware/`
- **AWS profile**: `REDACTED-PROFILE` (shared account 000000000000, role IibsAdminAccess-DO-NOT-DELETE)
- **SageMaker execution role**: `arn:aws:iam::000000000000:role/REDACTED-ROLE`
- **Instance quotas**: g5.xlarge on-demand=1 (shared), g5.4xlarge on-demand=30 (use this), g5.xlarge spot=5
- **Midway auth**: credentials expire every ~10h; run `mwinit` to refresh before launching jobs

## Checkpoint status (as of 2026-04-22)

| Dataset | RQ-VAE ckpt | Validation verdict | Decoder MTL ckpt |
|---------|------------|--------|------------------|
| Beauty  | `s3://REDACTED-BUCKET/rqvae-level-aware/checkpoints/rqvae_amazon_beauty/checkpoint_399999.pt` | ✅ HEALTHY (validated 2026-04-21): 248–256/256 unique SIDs per level, entropy 7.66–7.73 bits. Saved `model_config` shows `ROTATION_TRICK + decoder.normalize=True + n_cat_feats=0`. | ✅ `s3://REDACTED-BUCKET/rqvae-level-aware/decoder-mtl/beauty/decoder-mtl-beauty-od-20260412-2036/output/model.tar.gz` — trained against this RQ-VAE; MTL+eval numbers remain valid. |
| Sports (pre-fork) | `trained_models/rqvae_amazon_sports/checkpoint_high_entropy.pt` | ❌ Collapsed under prior fork code | ⚠️ `decoder-mtl-sports-od4-20260415-1851` — invalidated; retrain once a healthy Sports RQ-VAE exists
| Sports v6 / v7 | — | ❌ Collapsed under the since-reverted commit `57808c5` (L2-fix). Kept in S3 for reference only. | — |
| Beauty / Sports / Toys repro (2026-04-22) | `s3://REDACTED-BUCKET/rqvae-level-aware/rqvae/{beauty,sports,toys}-repro/.../model.tar.gz` | ❌ Collapsed. Trained under the since-reverted commit `0449747` (`kmeans_initted` buffer) which interacted badly with `@torch.compile` on `RqVae.forward` — the compiled graph re-ran KMeans on every forward pass, resetting the codebook. Do **not** use. | — |
| Beauty / Sports / Toys / Steam repro3 (2026-04-22 pm) | `s3://REDACTED-BUCKET/rqvae-level-aware/rqvae/{beauty,sports,toys,steam}-repro3/...` | ❌ All four validated as COLLAPSED (1 unique SID per level, entropy 0 bits, min_dist 0). Training curves look healthy up to ~step 5k (`vl` peaks around 0.05), then oscillate between steps 5k–12k and permanently settle to `vl=0` afterwards. Pattern is optimizer instability with ROTATION_TRICK + commitment_weight=0.25 on this data, not a load-time bug. The pre-fork healthy Beauty checkpoint used the same gin architecture, so the open question is what code change between then and now destabilised the optimization. | — |
| ML1M    | `trained_models/rqvae_ml1m/checkpoint_399999.pt` | — | ❌ Incompatible data pipeline (different feature dims, split structure, max_seq_len) — **dropped** |

Open questions after the repro3 round:

1. Training curves do not match the pre-fork healthy run: healthy training should hold a
   non-zero VQ loss across the entire horizon, not oscillate and collapse by step ~10k.
2. Reverting the L2-norm fix (`f9b645a`) and the kmeans_initted buffer (`4e0eb00`) was
   necessary — both commits reliably caused collapse — but neither is sufficient to get
   training healthy again.
3. `@torch.compile(mode="reduce-overhead")` on `RqVae.forward` is present in both the
   healthy era and today, so that's not the likely differentiator, but we have not
   directly tested disabling it.

Next actions (in priority order):

1. Bisect between `15126b7` (Apr 7 SageMaker wire-up, close to healthy-ckpt era) and
   `HEAD` with a **single-variable test**: 5k training steps on Beauty against the
   existing healthy gin config, compare VQ-loss curve. Do **not** chase multiple
   variables in parallel again. See `docs/bisect_runbook.md` for the launch commands.
2. In parallel, run one full training with `@torch.compile` disabled — if it validates
   HEALTHY, we have a `torch.compile` + accelerate + ROTATION_TRICK interaction we
   cannot safely keep.
3. As a safety net, try `commitment_weight=0.1` — the 0.25 → 0.05 → 0 oscillation is
   consistent with commitment loss overwhelming reconstruction once the encoder is
   close enough to the codebook.
4. Do **not** launch decoder-MTL / alpha-search / eval sweeps until at least one of
   Beauty / Sports / Toys / Steam validates healthy at end-of-training.

### Bisect levers wired into the training loop

All three planned experiments are selectable without editing source:

- `train.vae_mlp_activation="silu"|"relu"|"gelu"` and `train.vae_decoder_normalize=True|False`
  in the gin config. Together they can revert `a5367ed` (the only non-reverted commit
  in the RQ-VAE training path since the healthy era). The saved `model_config` does
  capture these going forward so validator reconstruction is faithful.
- `RQVAE_DISABLE_COMPILE=1` as a container env var → `modules.rqvae._maybe_compile`
  becomes an identity; `RqVae.forward` runs eagerly. Exposed on the two launchers as
  `--disable-compile`.
- `train.commitment_weight=0.1` (already gin-configurable).
- Per-level training logs now include `residual_avg_norm_{0,1,2}` alongside the
  existing `emb_avg_norm_{i}`. Watch for `||x||` shrinking or `||emb||` exploding in
  the 5k window before VQ loss zeroes — that discriminates rotation-scale blow-up from
  commitment-dominance.

Pre-built Beauty bisect configs live next to the dataset config:

- `configs/rqvae_amazon_beauty_bisect_a5367ed.gin` — ReLU + `decoder_normalize=False`.
- `configs/rqvae_amazon_beauty_bisect_cw01.gin` — `commitment_weight=0.1`.

## Datasets (candidates for paper)

- **Amazon Beauty**: ~22K users, ~12K items, auto-download. Known-healthy pre-fork RQ-VAE.
- **Amazon Sports**: ~35K users, ~18K items, auto-download. Retraining in progress.
- **Amazon Toys**: ~19K users, ~12K items, auto-download. Retraining in progress.
- **Steam**: ~334K users, ~13K items, HuggingFace download. Retraining in progress.

## Data prep

- Amazon (Beauty, Sports, Toys): auto-download via `AmazonReviews.download()` (single Google-Drive zip covers all three splits). Preprocessed features cached at `s3://REDACTED-BUCKET/rqvae-level-aware/datasets/amazon/` for Beauty + Sports + Toys.
- Steam: downloads from HuggingFace (UCSD mirror is 404). Processed cache at `s3://REDACTED-BUCKET/rqvae-level-aware/datasets/steam/`.
- (Re-)populate caches with `sagemaker/launch/launch_preprocess_datasets.py --splits <...>` then `--sync <job-name>`.
- Training jobs that mount `s3://.../datasets/<name>/` as a `dataset` channel skip the ~30-60 min preprocessing step — `override_save_dir_for_sagemaker()` symlinks `SM_CHANNEL_DATASET` onto the gin-configured `dataset_folder`.
- All datasets use leave-one-out split; 5-core filtering for Steam.
- Sentence-T5 features are already L2-normalized (verified Beauty + Sports: norm 0.9995–1.0005).

## Known issues and workarounds

### RQ-VAE training — lessons learned (2026-04-22)

Two earlier "fix" commits that produced the dropping of Sports/Steam/Toys
have been reverted after independent sanity runs proved they were the
cause, not the cure:

1. **The L2-norm "fix" (commit `57808c5`, reverted in `f9b645a`)**
   hypothesized that `torch.cat([l2norm(x_hat[..., :-0]), x_hat[..., -0:]])`
   returns un-normalized `x_hat` for `n_cat_feats=0`. That is true at the
   `torch.cat` level — but `self.decoder` is built with `normalize=True`,
   so `x_hat` arrives already L2-normalized from the decoder MLP. Adding
   an extra `l2norm(x_hat)` in an `else` branch introduced a double
   normalization that altered autograd gradients enough to destabilise
   Sports training (v6 and v7 both collapsed with all 3 levels pinned to
   a single SID). Upstream EdoardoBotta/RQ-VAE-Recommender uses the
   original `torch.cat` and trains cleanly on Amazon Beauty+Sports.

2. **The `kmeans_initted` buffer change (commit `0449747`, reverted in
   `4e0eb00`)** moved the Quantize layer's `kmeans_initted` from a Python
   bool to a tensor buffer so it would persist across `load_state_dict`.
   The buffer value gets captured at compile time by the
   `@torch.compile(mode="reduce-overhead")` decorator on `RqVae.forward`
   (CUDA-graph mode specializes the compiled graph on guard values).
   In-place `kmeans_initted.fill_(True)` mutates tensor data, not the
   captured Python value, so the compiled graph re-enters
   `Quantize._kmeans_init(x=current_batch)` on **every** forward pass —
   training effectively resets the codebook each iteration. All three
   repro runs (Beauty+Sports+Toys, 400k steps each) finished with
   `rl≈0.0013 vl=0.0000` and validated as completely collapsed.
   The bool version works because attribute mutation is a recompile
   trigger for `torch.compile`: after `_kmeans_init` sets
   `self.kmeans_initted = True`, the next forward re-traces under a new
   constant and takes the non-init branch.

Load-time handling: the real problem the buffer commit tried to solve
is legitimate — a loaded bool-attribute reverts to `False`, and the
next forward call overwrites the trained codebook. It is handled today
by setting `layer.do_kmeans_init = False` immediately after
`load_state_dict`. `evaluate/validate_rqvae.py` does this; downstream
eval / alpha-search scripts must do the same when they load the RQ-VAE
manually (most already route through `SemanticIdTokenizer` which passes
`codebook_kmeans_init=False`).

### Beauty's actual architecture

Beauty's healthy `checkpoint_399999.pt` was trained with
`QuantizeForwardMode.ROTATION_TRICK` + `decoder.normalize=True`
(verified from the saved `model_config`), not with `STE` as
`configs/rqvae_amazon_beauty.gin` previously claimed. The four active
configs now read `ROTATION_TRICK`.

### SageMaker dependency pins (requirements.txt)
These are critical — relaxing them breaks SageMaker containers:
- `polars==1.9.0` — newer versions change `.list.to_array()` return type from List to Array, breaking `_df_to_tensor_dict`
- `wandb>=0.19.0,<0.25` — wandb 0.25+ requires protobuf>=6.32.1 but SM container has 6.31.1; crashes at import time before `WANDB_MODE=disabled` takes effect
- `gin_config==0.5.0` — exact pin required
- Do NOT pin `triton` — SM container ships the correct version for its torch build; overriding it breaks `torch._inductor`

### SageMaker runtime
- `override_save_dir_for_sagemaker()` in `modules/utils.py` redirects output to `/opt/ml/model/` and disables wandb when no API key is present
- `parse_config()` accepts both positional args (local) and `--config_path` flag (SageMaker)
- `source_dir="."` uploads the entire repo including `trained_models/` — keep only final checkpoints there
- Spot instances on g5.xlarge/g5.2xlarge are unreliable (frequent interruptions, no checkpointing); use **g5.4xlarge on-demand** (30 instance quota)

### Toys dataset
repro3 run (`rqvae-toys-repro3-20260422-113756`) validated COLLAPSED — see repro3 row above. Prior drop notes listing "KMeans init + STE mode on this particular data distribution" as the cause were based on collapses produced by the since-reverted `0449747` buffer commit. Toys never actually exhibited architecture-specific incompatibility with this codebase — re-assess once the fresh training completes and validates.

### ML1M dataset
Dropped due to pipeline incompatibilities: different feature dims (786 vs 768), missing `is_train`/`text` fields in data loader, different split structure (`eval` vs `test`), `max_seq_len=200` vs 20, CUDA index-out-of-bounds from padding. Would require significant data pipeline refactoring.

## Pipeline stages

```
RQ-VAE training: repro3 jobs all collapsed; root-causing before next round
  ↓ (gated on end-of-training validation via evaluate/validate_rqvae.py)
Decoder baseline + MTL training (per dataset) — train_decoder.py / train_decoder_mtl.py
  ↓
Alpha grid search (per dataset) — scripts/alpha_grid_search.py
Alpha learned (per dataset) — scripts/train_alpha_params.py
  ↓
Eval sweep (8 strategies × N datasets) — evaluate/run_eval.py
Residual entropy analysis — modules/analysis/residual_entropy.py
  ↓
Collect results → figures/tables → paper
```

## Test suite

```bash
pytest tests/ -q                    # 92 tests across 12 test files
pytest tests/ --cov --cov-fail-under=80
```

Known gaps: `tests/integration/` is empty, no `tests/data/test_steam_loader.py`.

## Decoding strategy registry

Strategies registered in `modules/decoding/__init__.py`:
`vanilla`, `dbs`, `gumbel_topk`, `hybrid`, `level_aware_mix`, `level_aware_mix_grid`, `level_aware_mix_learned`, `sasrec_rerank`

## Keeping this file up to date

Update AGENTS.md whenever:
- A training job completes or is dropped — update the checkpoint status table
- A new dataset or config is added or removed
- A dependency pin changes in `requirements.txt` (add to known issues if SageMaker-related)
- A new decoding strategy is registered in `modules/decoding/__init__.py`
- Pipeline stages change status (training → alpha search → eval → paper)
- A new workaround or known issue is discovered

The checkpoint status table and pipeline stages section are the most frequently stale — check them first.
