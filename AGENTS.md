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

## Checkpoint status (as of 2026-04-16)

| Dataset | RQ-VAE ckpt | Decoder MTL ckpt |
|---------|------------|------------------|
| Beauty  | `trained_models/rqvae_amazon_beauty/checkpoint_399999.pt` | ✅ `s3://REDACTED-BUCKET/rqvae-level-aware/decoder-mtl/beauty/decoder-mtl-beauty-od-20260412-2036/output/model.tar.gz` |
| Sports  | `trained_models/rqvae_amazon_sports/checkpoint_high_entropy.pt` | ✅ `s3://REDACTED-BUCKET/rqvae-level-aware/decoder-mtl/sports/decoder-mtl-sports-od4-20260415-1851/output/model.tar.gz` |
| Steam   | `trained_models/rqvae_steam/checkpoint_399999.pt` | 🔄 `decoder-mtl-steam-od4-20260415-1851` (in progress, ~65% at 20h) |
| Toys    | ⚠️ Codebook collapse (SID=0, all metrics=1.0) — **dropped** | — |
| ML1M    | `trained_models/rqvae_ml1m/checkpoint_399999.pt` | ❌ Incompatible data pipeline (different feature dims, split structure, max_seq_len) — **dropped** |

## Datasets (final set for paper)

- **Amazon Beauty**: ~22K users, ~12K items, auto-download
- **Amazon Sports**: ~35K users, ~18K items, auto-download
- **Steam**: ~334K users, ~13K items after 5-core filtering; downloads from HuggingFace mirror (`recommender-system/steam-review-and-bundle-dataset`)

## Data prep

- Amazon (Beauty, Sports): auto-download via existing loaders
- Steam: downloads from HuggingFace (UCSD mirror is 404); first run `python -c "from data.steam import RawSteam; RawSteam().download()"` to cache raw data
- All datasets use leave-one-out split; 5-core filtering for Steam

## Known issues and workarounds

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
Dropped due to RQ-VAE codebook collapse: VQ loss=0.0 throughout training, inverted variance pattern (level 0 lowest, level 2 highest), decoder SID loss=0.0, all eval metrics=1.0. Root cause: KMeans init + STE mode on this particular data distribution.

### ML1M dataset
Dropped due to pipeline incompatibilities: different feature dims (786 vs 768), missing `is_train`/`text` fields in data loader, different split structure (`eval` vs `test`), `max_seq_len=200` vs 20, CUDA index-out-of-bounds from padding. Would require significant data pipeline refactoring.

## Pipeline stages (post-training)

```
Decoder MTL complete (Beauty ✅, Sports ✅, Steam 🔄)
  ├── Alpha grid search (per dataset) — scripts/alpha_grid_search.py
  ├── Alpha learned (per dataset) — scripts/train_alpha_params.py
  ├── Eval sweep (8 strategies × 3 datasets) — evaluate/run_eval.py
  ├── Residual entropy analysis — modules/analysis/residual_entropy.py
  └── Collect results → figures/tables → paper
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
