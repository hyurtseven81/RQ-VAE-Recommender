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

## Module layout

```
modules/decoding/      — beam search strategies (vanilla, dbs, gumbel, hybrid, level_aware_mix, sasrec_reranker)
modules/heads/         — sasrec_head.py, mtl_losses.py
modules/analysis/      — residual_entropy.py
evaluate/              — metrics.py, stats.py, result_store.py
scripts/               — alpha_grid_search.py, train_alpha_params.py, make_figures.py, make_tables.py
sagemaker/             — launch scripts for SageMaker training jobs
paper/                 — LaTeX source (CIKM 2026 submission)
```

## AWS / S3

- **Bucket**: `s3://REDACTED-BUCKET/rqvae-level-aware/`
- **AWS profile**: `REDACTED-PROFILE` (compute machine only — not available locally)
- Upload datasets: `python sagemaker/setup/upload_datasets.py`
- Upload checkpoints: manually via `aws s3 cp trained_models/ s3://REDACTED-BUCKET/rqvae-level-aware/checkpoints/ --recursive --profile REDACTED-PROFILE`

## Checkpoint status (as of 2026-04-07)

| Dataset | RQ-VAE ckpt | Decoder ckpt |
|---------|------------|--------------|
| Beauty  | `trained_models/rqvae_amazon_beauty/checkpoint_399999.pt` | needs training |
| Sports  | `trained_models/rqvae_amazon_sports/checkpoint_high_entropy.pt` | needs training |
| Toys    | **missing** — must train `python train_rqvae.py configs/rqvae_amazon.gin` (set dataset=toys) | needs training |
| Steam   | **missing** — must train `python train_rqvae.py configs/rqvae_steam.gin` | needs training |

## Data prep

- Amazon (Beauty, Sports, Toys): auto-download via existing loaders, no manual step needed
- Steam: first run `python -c "from data.steam import RawSteam; RawSteam().download()"` to cache raw data
- All datasets use leave-one-out split; 5-core filtering for Steam

## Test suite

```bash
pytest tests/ -q                    # 92 tests across 12 test files
pytest tests/ --cov --cov-fail-under=80
```

Known gaps: `tests/integration/` is empty (no end-to-end MTL test yet), no `tests/data/test_steam_loader.py`.

## Decoding strategy registry

Strategies registered in `modules/decoding/__init__.py`:
`vanilla`, `dbs`, `gumbel_topk`, `hybrid`, `level_aware_mix`, `level_aware_mix_grid`, `level_aware_mix_learned`, `sasrec_rerank`
