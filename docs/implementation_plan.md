# Level-Aware Hybrid Decoding for Generative Retrieval — Implementation Plan

**Working paper title:** "Level-Aware Hybrid Decoding for Generative Retrieval: Mixing Dense and Autoregressive Scores at Each Codebook Level"

**Date drafted:** 2026-04-07

---

## 1. Codebase Findings

### 1.1 Beam Search: Location, Mechanics, and Hook Points

**Function:** `generate()` in `modules/model.py:301–391`

**Critical finding — this is NOT standard beam search.** It is a sampling-based approach:
1. For each codebook level `h`, calls `torch.multinomial(probas, num_samples=n_cands)` where `n_cands = min(64, num_embeddings_per_hierarchy)` (line 314).
2. Validates sampled candidates via `_check_valid_prefix()` (lines 349–370), which checks against a pre-registered `codebooks` buffer.
3. Masks invalid prefixes with `float("-inf")` and keeps top-k by cumulative log-probability.
4. Maintains a KV cache (`EncoderDecoderCache`) for efficient generation and reorders it after beam pruning (line 380).

This design means DBS and Gumbel top-k will replace the multinomial sampling step (line 345), not add a penalty on top of softmax output. The valid-prefix mask is applied *after* sampling/perturbation and must remain in place for every strategy.

**Default beam size:** `top_k_for_generation=10` (`modules/model.py:66`, `configs/decoder_amazon.gin:27`)

**Trie validity function:** `_check_valid_prefix()` at `modules/model.py:169–182`. Takes `prefix: [N, depth]`, checks each row against the registered `self.codebooks` buffer (registered at `__init__` line 75). Returns a boolean mask `[N]`. Batches internally in chunks of 100,000. **Every new strategy must route through this function — the interface must make it impossible to bypass.**

**Codebook dimensions for Amazon Beauty (from `configs/decoder_amazon.gin`):**
- Levels `L = 3` (configured as `vae_n_layers=3`)
- Codebook size per level: 256 (`vae_codebook_size=256`)
- Codebook embedding dim: 32 (`vae_embed_dim=32`)
- Decoder d_model: 384 (`t5_d_model=384`)

**Are codebook embeddings accessible inside the beam loop?** Yes. `rqvae.layers[l]` is a `Quantize` module whose `nn.Embedding` table (`embed.weight`) has shape `[256, 32]`. The RQ-VAE is frozen and registered in the decoder model at init time (line 75 registers `codebooks` buffer, but the actual Quantize modules are on the RQ-VAE object). The implementing agent must ensure the codebook embedding tensors are pre-fetched and available to the decoding strategies without recomputation — pass them as an argument at strategy construction time.

**Can we compute partial RQ reconstruction `Σ_{i≤l} e_{c_i}` cheaply?** Yes. During beam expansion at level `l`, we already have the sequence of chosen tokens for levels `0..l-1`. We look up `rqvae.layers[i].embed.weight[c_i]` for each `i < l` and sum them. This is one gather + sum per beam, fully batched. The embeddings live in 32-d space so cost is negligible.

**Decoder hidden state shape before SID heads:** At `modules/model.py:288`:
```python
decoder_output = self.decoder_forward_pass(...)[:, :-1]  # [B, num_hierarchies, d_model]
```
For Amazon configs: shape = `[B, 3, 384]`. The per-level hidden state at position `h` is `decoder_output[:, h, :]` shape `[B, 384]`. This is what feeds `self.decoder_mlp[h]` (a `Linear(384, 256)`). The SASRec head will consume the **final** position hidden state: `decoder_output[:, -1, :]` shape `[B, 384]` — taken after all L SID tokens have been processed in teacher forcing, capturing the full sequence representation.

During generation (not teacher forcing), the equivalent is the hidden state from the decoder at the last generation step. In `generate()`, `dec_out[:, -1, :]` at the last hierarchy level `h = L-1` is the right tensor.

### 1.2 RQ-VAE Codebook Access

**File:** `modules/rqvae.py:60–138`

Codebook layers: `self.layers = nn.ModuleList([Quantize(...) for _ in range(n_layers)])` at line 64. Each `Quantize` has `self.embed: nn.Embedding(n_embed, embed_dim)` (in `modules/quantize.py`).

Residuals per level: `RqVaeOutput.residuals` shape after `rearrange(residuals, "b h d -> h d b")` is `[L, D, B]` = `[3, 32, B]`. The residual at level `l` before quantization is `residuals[l]` — this is what the residual entropy analysis (§5.8) computes norms over.

### 1.3 Evaluation

**File:** `evaluate/metrics.py:1–26`

Current `TopKAccumulator`:
- Computes **Recall@K only** (labeled `h@{k}` in code). **NDCG is not implemented.**
- Accumulates across all samples — **no per-user tracking.**
- Returns a single float per K value.

**What needs to be added:**
1. Per-user metric arrays (store `{user_id: match_vector}` dict).
2. NDCG@K computation.
3. Paired bootstrap testing (new module `evaluate/stats.py`).
4. Parquet result store.

### 1.4 Checkpoints Inventory

```
trained_models/
├── rqvae_amazon_beauty/
│   ├── checkpoint_399999.pt          ← main Beauty RQ-VAE checkpoint
│   ├── checkpoint_high_entropy.pt    ← high-entropy variant
│   └── checkpoint_high_entropy_ste.pt
├── rqvae_amazon_sports/
│   └── checkpoint_high_entropy.pt    ← Sports RQ-VAE checkpoint
└── rqvae_ml32m/
    └── checkpoint_high_entropy.pt    ← ML32M (not needed for this project)
```

**Missing checkpoints:**
- **Amazon Toys RQ-VAE** — must train once on SageMaker before any Toys experiment.
- **Steam RQ-VAE** — must train once on SageMaker after Steam preprocessing.
- No decoder checkpoints exist — all decoder training is new.

**No Toys checkpoint is the first blocking dependency.** Plan assumes `trained_models/rqvae_amazon_toys/checkpoint_high_entropy.pt` will exist after PR 4/5.

### 1.5 Dataset Loaders

**Supported:** `RecDataset.AMAZON` (Beauty/Sports/Toys via `dataset_split` param), `ML_1M`, `ML_32M` (`data/processed.py:19–22`).

**Not supported:** Steam. A new `data/steam.py` loader must be written following LIGER's preprocessing conventions (see §6).

### 1.6 Existing Gumbel Code

`distributions/gumbel.py` implements `sample_gumbel()` (line 8: `−log(−log(U + eps))`) and `gumbel_softmax_sample()`. The `sample_gumbel` function is exactly what Gumbel top-k decoding needs. **Reuse this; do not duplicate.**

### 1.7 LIGER Compatibility

LIGER (`facebookresearch/liger`) uses a **T5 encoder-decoder** backbone with:
- An SID head (next-token cross-entropy, equivalent to our decoder).
- An **embedding head** projecting to frozen Sentence-T5 item embeddings (content signal — NOT collaborative signal).
- Inference: beam search produces K candidates from the SID head → cold-start augmentation → Sentence-T5 dense head reranks the augmented pool. **This is post-hoc reranking, not per-level mixing.**

LIGER's transformers version may differ from ours (our lock has `transformers` from sentence_transformers). A separate conda environment or Docker image is required for LIGER. **Do not import LIGER code into this fork.**

### 1.8 Open Issues / Contradictions

1. **Sampling vs. beam search framing:** The current `generate()` samples candidates then keeps top-k — it's not greedy beam search. DBS and Gumbel strategies must be described as modifying the *sampling distribution*, not adding penalties to logit vectors. The paper should frame this accurately: "we modify the candidate sampling step."

2. **n_cands=64 vs. codebook_size=256:** Only 64 of 256 candidates are sampled per step. This means some valid items are not explored each run. Increasing `n_cands` (up to 256) is a cheap hyperparameter that should be swept in the beam size sensitivity experiment.

3. **No NDCG implementation exists.** NDCG must be added to `evaluate/metrics.py` in PR 2.

4. **Toys RQ-VAE checkpoint absent.** Cannot train Toys decoder until this is available. PR 4 (SageMaker) launches this job; PR 5 waits for it.

5. **Codebook embedding dimension (32) << decoder d_model (384).** The SASRec auxiliary head's item embedding space (`D_item`) will be 32 (matching RQ-VAE encoder output) or 384 (matching decoder). The RQ-VAE reconstruction operates in 32-d space, so the per-level mixing dot product `<q_SASRec, r_l^cand>` should happen in 32-d. The SASRec head MLP must project from 384 → 32 (not 384 → 384) to align with the codebook embedding space for per-level mixing.

---

## 2. Prior Art Summary

| Paper | Summary | Differentiation from Our Work |
|---|---|---|
| **TIGER** (Rajput et al., NeurIPS 2023) | Introduces semantic IDs via RQ-VAE tokenization + T5 seq2seq for generative recommendation. This fork (`EdoardoBotta/RQ-VAE-Recommender`) is a clean reimplementation. | Our base model; we extend its decoding, not its architecture. |
| **LIGER** (Yang et al., arXiv:2411.18814, Nov 2024) | Adds a frozen Sentence-T5 content-embedding head to TIGER; post-hoc reranks top-K beams with dense scores after full generation. | We mix scores *during* beam expansion at each codebook level. LIGER reranks after; we mix per-level within. Also: LIGER uses frozen text embeddings (content signal); we use learned RQ-VAE encoder embeddings (collaborative signal). |
| **COBRA** (Yang et al., arXiv:2503.02453, NeurIPS 2025) | BeamFusion: combines autoregressive + sparse-dense scores after all codebook tokens are decoded, before final ranking. Cascaded sparse-dense. | COBRA fuses after full item generation (BeamFusion over completed beams). We fuse at each codebook level during expansion. BeamFusion is post-generation; per-level mixing is in-loop. |
| **EAGER** (Wang et al., arXiv:2406.14017, KDD 2024) | Two decoders: behavioral stream (interaction IDs) + semantic stream (text embeddings), fused at inference. | Architecture-level fusion (two decoders); we are decoding-time fusion within a single decoder. Related work only. |
| **GD-RIPOR** (Zeng et al., arXiv:2404.14600) | Dense-guided constrained beam search for document retrieval: uses dense scores to constrain beam expansion. | Document retrieval (not recommendation), no per-level weighting, no RQ-VAE structure. Closest prior art on "dense-guided decoding" angle. |
| **Penha et al. 2024** (RecSys 2024, arXiv:2410.16823) | Off-the-shelf Diverse Beam Search applied to TIGER for recommendation. Standard Hamming-style diversity penalty. | Uses token-ID diversity (Hamming); we use codebook-embedding-distance diversity (L2 in embedding space at each level). |
| **GRID handbook** (Ju et al. 2025, arXiv:2507.22224) | Practitioner's guide to generative retrieval with semantic IDs; ablations on beam size, codebook size, etc. | Background/context for experiment design; no methodological overlap. |
| **SASRec** (Kang & McAuley 2018) | Self-attention sequential model; state-of-the-art dense sequential recommender. | Provides the design pattern for our auxiliary head (self-attention over history → item scoring), not used directly but our head is "SASRec-style." |
| **Diverse Beam Search** (Vijayakumar et al. 2016) | DBS: partition beams into groups, add diversity penalty for intra-group similarity. | Standard DBS uses token-level penalty; our codebook-embedding-distance variant is unpublished in RecSys or IR contexts. |
| **Stochastic Beam Search** (Kool et al. 2019, ICML) | Gumbel top-k: perturb log-probs with Gumbel noise, take top-k. Provides unbiased samples without replacement from the beam distribution. | Never applied to recommendation generative retrieval. We implement this and validate coverage/diversity properties. |
| **VALL-E / AudioLM / SoundStorm** | Speech synthesis models using RQ-codecs with per-level architectural differentiation (AR for level-1, NAR for levels 2+; separate models per level group; MaskGIT iterations proportional to level). | These treat levels differently via architecture at training time. We treat levels differently via score mixing at inference time. The speech literature establishes that per-level treatment of RQ hierarchies is principled; our contribution is bringing this to decoding via score mixing. |

**The unclaimed position:** No published work performs per-codebook-level mixing of generative and dense scores during beam expansion for any retrieval task.

---

## 3. Repository Setup Tasks

### Task 3.1 — Branch structure

**What:** Create feature branches off `develop`.
**Why:** Clean PR history; each branch maps to one section of the paper contribution.

```
main       ← upstream tracking, never directly committed
develop    ← integration branch
feat/eval-harness
feat/sagemaker-infra
feat/liger-baseline
feat/decoding-strategies
feat/sasrec-head
feat/level-aware-mixing
feat/residual-analysis
```

**Files:** `.git/` (branch creation only)
**Tests:** None
**Effort:** 0.5h
**Dependencies:** None

### Task 3.2 — Upstream remote + CONTRIBUTING.md

**What:** Add `upstream` remote pointing to `EdoardoBotta/RQ-VAE-Recommender`; document rebase workflow in `CONTRIBUTING.md`.
**Why:** Keep fork synchronized with upstream improvements during the 47-day sprint.

**Files created:** `CONTRIBUTING.md`
**Tests:** None
**Effort:** 0.5h

### Task 3.3 — Dependency management

**What:** Add `pyproject.toml` with pinned versions for all new dependencies. Keep existing `requirements.txt` for backward compat.
**Why:** Reproducible environments for CI and SageMaker.

**New deps to pin:** `scipy>=1.13`, `optuna>=3.6`, `pyarrow>=16.0`, `pandas>=2.0`, `matplotlib>=3.9`, `seaborn>=0.13`, `pytest>=8.0`, `pytest-cov>=5.0`, `ruff>=0.4`, `mypy>=1.10`, `boto3>=1.34`, `sagemaker>=2.220`, `pre-commit>=3.7`

**Files created:** `pyproject.toml`
**Effort:** 1h

### Task 3.4 — LIGER sibling environment

**What:** Create `../liger-env/` conda env spec file at `envs/liger_environment.yml` (not inside the fork, just the spec). Document the sibling directory convention.
**Why:** LIGER may need different PyTorch/transformers versions. Must not pollute our training environment.

**Files created:** `envs/liger_environment.yml`, note in `CONTRIBUTING.md`
**Effort:** 1h (verify after cloning LIGER)

### Task 3.5 — CI pipeline

**What:** `.github/workflows/ci.yml` — ruff lint, mypy on `modules/decoding/`, `modules/heads/`, `evaluate/`, `sagemaker/`; pytest with 80% coverage gate on new code.
**Why:** Catch regressions before every merge to `develop`.

**Files created:** `.github/workflows/ci.yml`, `.pre-commit-config.yaml`
**Effort:** 1.5h

### Task 3.6 — README and LICENSE notice

**What:** Add "Research fork" section at top of README; add derivative-work notice to LICENSE.
**Effort:** 0.5h

---

## 4. Decoding Framework Implementation

### Task 4.1 — Module skeleton

**What:** Create `modules/decoding/__init__.py` with a registry dict `DECODING_STRATEGIES`.
**Why:** Gin-selectable strategy pattern; new strategies can be added without touching the model.

**Files created:** `modules/decoding/__init__.py`
**Tests:** None (pure registry)
**Effort:** 0.5h

### Task 4.2 — Abstract base strategy

**What:** `modules/decoding/base.py` — `BeamSearchStrategy` ABC with:
```python
def expand(
    self,
    beams: Tensor,           # [B, k, h] current beams
    log_probas: Tensor,      # [B, k] cumulative log-probs
    h: int,                  # current codebook level
    probas: Tensor,          # [B*k, vocab] softmax probas from decoder
    check_valid_fn: Callable, # _check_valid_prefix bound method — CANNOT be bypassed
    codebook_embs: List[Tensor],  # List[L] of [vocab, 32] codebook embedding tensors
    decoder_hidden: Optional[Tensor] = None,  # [B*k, d_model] for hybrid strategies
) -> Tuple[Tensor, Tensor, Any]:  # new_beams, new_log_probas, kv_reorder_indices
```
Return value `kv_reorder_indices` tells the caller how to reorder the KV cache; vanilla strategy returns `parent_beam_idx` as today.

**Files created:** `modules/decoding/base.py`
**Tests:** None (ABC)
**Effort:** 1h
**Dependencies:** Task 4.1

### Task 4.3 — Vanilla strategy refactor

**What:** `modules/decoding/vanilla.py` — extract the existing beam search logic from `generate()` (lines 328–389 of `model.py`) into a `VanillaBeamSearch(BeamSearchStrategy)` class. The `generate()` method becomes a thin loop that calls `strategy.expand()`. Must be byte-for-byte equivalent to the original.
**Why:** This is the regression-gating refactor. If this breaks, nothing else can land.

**Files modified:** `modules/model.py` (thin loop), `modules/decoding/vanilla.py` (extracted logic)
**Tests:**
- `tests/decoding/test_vanilla_regression.py` — load a saved fixture of beam outputs from the pre-refactor `generate()`, run through refactored path, assert identical top-k beams and log_probas.
**Effort:** 3h
**Dependencies:** Task 4.2
**Gating:** The regression test must pass before any other decoding strategy is merged.

### Task 4.4 — Codebook-aware Diverse Beam Search

**What:** `modules/decoding/dbs.py` — `DiverseBeamSearch(BeamSearchStrategy)`.

Algorithm:
1. Divide `k` beams into `G = k // 2` groups of 2 (configurable).
2. For group `g`, compute logits as usual. Add penalty:
   ```
   penalty(cand) = λ_l · Σ_{g' < g} min_over_chosen_g' L2(emb_cand_l, emb_chosen_g'_l)
   ```
   where `emb_cand_l = codebook_embs[h][cand_token_id]` (shape `[32]`).
3. Add penalty to log-probs **before** validity masking.
4. Apply validity mask, sample/sort, keep top-k/G per group.
5. If a group has 0 valid completions after masking: fall back to vanilla for that group, increment a warning counter.

Config: `lambda_per_level: List[float]` (length L), `num_groups: int`

**Files created:** `modules/decoding/dbs.py`
**Tests:**
- `tests/decoding/test_dbs.py` — synthetic 2-level, 4-token codebook; verify diversity increases with λ.
- `tests/decoding/test_dbs.py::test_diversity_monotone_in_lambda`
- `tests/decoding/test_trie_mask_compose.py::test_dbs_all_valid`
**Effort:** 4h
**Dependencies:** Task 4.3

### Task 4.5 — Gumbel top-k decoding

**What:** `modules/decoding/gumbel.py` — `GumbelTopKBeamSearch(BeamSearchStrategy)`.

Algorithm: Replace `torch.multinomial(probas, n_cands)` with:
```python
perturbed = torch.log(probas) / tau + sample_gumbel(probas.shape, probas.device)
samples = perturbed.topk(n_cands, dim=-1).indices
samp_log_p = torch.log(torch.gather(probas, 1, samples))  # true log-probs for scoring
```
Track true log-probs (not perturbed) for cumulative scoring so Recall@K is measured on true probabilities. Reuse `distributions.gumbel.sample_gumbel`.

Config: `tau: float = 1.0`

**Files created:** `modules/decoding/gumbel.py`
**Tests:**
- `tests/decoding/test_gumbel.py::test_unbiased_at_tau_one` — at τ=1, beam_size=1, marginal over 10,000 seeds matches true softmax within KL < 0.01.
- `tests/decoding/test_gumbel.py::test_seed_reproducibility`
- `tests/decoding/test_trie_mask_compose.py::test_gumbel_all_valid`
**Effort:** 3h
**Dependencies:** Task 4.3

### Task 4.6 — Hybrid (det + stoch) strategy

**What:** `modules/decoding/hybrid.py` — `HybridBeamSearch(BeamSearchStrategy)`. Runs vanilla for `k_det` slots, Gumbel for `k_stoch = k - k_det` slots, merges, deduplicates by SID tuple, keeps top-k.

Config: `k_det: int` (default: `k // 2`)

**Files created:** `modules/decoding/hybrid.py`
**Tests:** `tests/decoding/test_hybrid.py` — verify k_det slots come from deterministic top-k; no duplicates.
**Effort:** 2h
**Dependencies:** Tasks 4.4, 4.5

### Task 4.7 — Cross-strategy tests

**What:** Two shared test files covering invariants that all strategies must satisfy.
- `tests/decoding/test_trie_mask_compose.py` — parametrized over all strategies; asserts every returned beam is a valid prefix in the corpus codebook.
- `tests/decoding/test_beam_size_consistency.py` — returned beam count = requested k (or fewer only when the trie has genuinely fewer completions).

**Effort:** 2h
**Dependencies:** Tasks 4.3–4.6

---

## 5. Level-Aware Hybrid Decoding (Primary Contribution)

### Task 5.1 — SASRec auxiliary head

**What:** `modules/heads/sasrec_head.py` — `SASRecAuxHead(nn.Module)`.

Architecture:
```
Input: decoder_hidden [B, d_model=384]
→ LayerNorm(384)
→ Linear(384, 384, bias=True) + GELU
→ Dropout(0.1)
→ Linear(384, D_item=32)  ← output: query vector q ∈ R^32
```
`D_item = 32` to match the RQ-VAE codebook embedding space (from §1.8 finding: codebook dim is 32, not 384).

Item embedding table: `nn.Embedding(num_items, 32)` initialized at construction from the frozen RQ-VAE encoder outputs (`rqvae.encode(item_features)` for each item). This bootstrap is done once at init; the table is then learned jointly during MTL training.

**Explicit architectural distinction from LIGER:** LIGER's dense head projects decoder hidden state to frozen Sentence-T5 *text* embeddings (content signal, 768-d, never updated). Our head projects to a learned item embedding table initialized from RQ-VAE encoder outputs (collaborative signal, 32-d, refined via MTL). The signal source (behavioral vs. content) and the embedding dimensionality are both different. This differentiation must be stated explicitly in the paper.

Forward returns: `q` shape `[B, 32]`. Scoring: `scores = q @ item_embedding_table.weight.T` → `[B, num_items]`.

**Files created:** `modules/heads/__init__.py`, `modules/heads/sasrec_head.py`
**Tests:** `tests/heads/test_sasrec_head.py` — forward shapes, gradient flow, init from RQ-VAE.
**Effort:** 3h
**Dependencies:** Task 3.3 (deps)

### Task 5.2 — MTL training script

**What:** `train_decoder_mtl.py` — new entry point sharing a common `_train_loop(model, aux_head, config, ...)` helper with `train_decoder.py` to avoid divergence.

Loss:
```
L_total = L_sid + λ_aux(t) · L_sasrec
```
where:
- `L_sid` = sum of per-level cross-entropy losses (unchanged from current `train_decoder.py`)
- `L_sasrec` = sampled-softmax InfoNCE: positives = actual next item; negatives = in-batch + 1024 uniform random. Temperature: learnable scalar or fixed=0.05 (ablation).
- `λ_aux(t)` = linear warmup from 0 to `λ_aux_max` over first 10% of training steps. Default `λ_aux_max = 0.2`.

Gradient clipping: norm 1.0 (check against upstream — `train_decoder.py` does not currently clip gradients; document this deviation).

**Files created:** `train_decoder_mtl.py`
**Files modified:** `train_decoder.py` (extract shared `_train_loop`)
**Tests:** `tests/training/test_mtl_loss.py` — loss decomposition, λ warmup schedule, gradient flow to both heads.
**Effort:** 5h
**Dependencies:** Task 5.1

### Task 5.3 — MTL gin configs

**What:** Four new gin config files for MTL training.

| File | Dataset | RQ-VAE checkpoint |
|---|---|---|
| `configs/decoder_amazon_beauty_mtl.gin` | Beauty | `trained_models/rqvae_amazon_beauty/checkpoint_399999.pt` |
| `configs/decoder_amazon_sports_mtl.gin` | Sports | `trained_models/rqvae_amazon_sports/checkpoint_high_entropy.pt` |
| `configs/decoder_amazon_toys_mtl.gin` | Toys | `trained_models/rqvae_amazon_toys/checkpoint_high_entropy.pt` *(must be trained first)* |
| `configs/decoder_steam_mtl.gin` | Steam | `trained_models/rqvae_steam/checkpoint_high_entropy.pt` *(must be trained first)* |

**Files created:** 4 gin configs
**Effort:** 1h
**Dependencies:** Task 5.2, RQ-VAE checkpoints for Toys and Steam (from PR 4/5)

### Task 5.4 — Per-level mixing mechanism

**What:** `modules/decoding/level_aware_mix.py` — `LevelAwareHybridDecoding(BeamSearchStrategy)` that wraps any base strategy.

At each codebook level `h`, in `expand()`:
1. Get base strategy's logits (probas before masking) from the wrapped strategy.
2. Compute partial RQ reconstruction for each candidate:
   ```python
   # r_l^cand for each (beam, candidate) pair
   # beams so far: [B, k, h], candidate token ids: [B*k, n_cands]
   past_embs = sum(codebook_embs[i][beams[:,:,i]] for i in range(h))  # [B, k, 32]
   cand_embs = codebook_embs[h][candidate_ids]  # [B*k, n_cands, 32]
   r_l = past_embs.unsqueeze(2) + cand_embs  # [B, k, n_cands, 32]
   ```
3. Compute SASRec score: `s = r_l @ q.unsqueeze(-1)` where `q = aux_head(decoder_hidden)` shape `[B*k, 32]`. Result: `[B, k, n_cands]`.
4. Z-score normalize log-probs and SASRec scores across the `n_cands` dimension (Option A default).
5. Mix: `score = (1 - alpha[h]) * norm_log_p + alpha[h] * norm_s_sasrec`.
6. Proceed with validity masking and top-k as normal.

`alpha` parameter: `List[float]` of length L, passed as constructor argument. Can be fixed (grid search) or a `nn.Parameter` (learned variant).

**Note on composability:** Because `LevelAwareHybridDecoding` wraps a base strategy and calls `base_strategy.expand()` to get probas before modifying them, it composes automatically with DBS, Gumbel, and hybrid strategies. The paper explicitly demonstrates this composability as a design contribution.

**Files created:** `modules/decoding/level_aware_mix.py`
**Tests:**
- `tests/decoding/test_level_aware_compose.py` — wrap vanilla, DBS, Gumbel; assert trie mask still respected.
- `tests/decoding/test_alpha_zero_equiv.py` — alpha=[0,0,0] produces beams identical to base strategy. **This is the critical correctness test for score normalization.**
- `tests/decoding/test_alpha_one_uses_sasrec.py` — alpha=[1,1,1] on a synthetic example where SASRec scores determine ranking; verify order matches SASRec ordering.
**Effort:** 6h
**Dependencies:** Tasks 4.3, 5.1

### Task 5.5 — SASRec reranker (H2a — implement first, before in-loop mixing)

**What:** `modules/decoding/sasrec_reranker.py` — post-hoc reranker that runs after any base decoding strategy, not in-loop.

Takes top-K completed beams (full SID tuples) → reconstructs full item embedding `r_full = Σ_l e_{c_l}` → scores `(1-α) * norm_log_p(beam) + α * norm_s_sasrec(item)` → reranks. Sweeps α ∈ {0.0, 0.1, ..., 1.0}.

**Why implement first:** This is the ablation that isolates post-hoc fusion (LIGER-style conceptually) from per-level in-loop mixing. The delta between H2a and H2b-grid is the paper's headline result.

**Files created:** `modules/decoding/sasrec_reranker.py`
**Tests:** `tests/decoding/test_sasrec_reranker.py` — at α=0 produces same ranking as base strategy; at α=1 produces same ranking as pure SASRec.
**Effort:** 3h
**Dependencies:** Task 5.1, Task 4.3

### Task 5.6 — Alpha schedule: grid search + learned variant

**What:** Two mechanisms for setting the per-level alpha schedule.

**Grid search variant:**
- `scripts/alpha_grid_search.py` — coarse grid α_l ∈ {0.0, 0.25, 0.5, 0.75, 1.0} for L=3 levels → 5³ = 125 configs per dataset. Each config is inference-only (~minutes). Optuna TPE refinement: 100 trials, ±0.15 around coarse winner per level.
- Parallelized via SageMaker (one job per config on `ml.g5.xlarge` spot).

**Learned variant:**
- Add `alpha_params: nn.Parameter(torch.zeros(L))` to `LevelAwareHybridDecoding`.
- `sigmoid(alpha_params)` constrains to (0, 1).
- `scripts/train_alpha_params.py` — freeze decoder + aux head; train only `alpha_params` for 1 epoch minimizing InfoNCE loss using mixed scores on training set.

**Falsification prior (state before running):** "The optimal alpha schedule is monotone non-increasing in l (α_0 ≥ α_1 ≥ α_2). Motivation: residual norms decrease with depth, so coarse levels carry more information and should receive more dense guidance; fine levels are near-noise and should defer to the autoregressive prior."

**Files created:** `scripts/alpha_grid_search.py`, `scripts/train_alpha_params.py`
**Tests:** None (scripts); gin config validates alpha list length.
**Effort:** 4h
**Dependencies:** Task 5.4

### Task 5.7 — Residual entropy analysis

**What:** `modules/analysis/residual_entropy.py` — compute per-level residual statistics.

For each dataset's trained MTL decoder:
1. Iterate all items, call `rqvae.get_semantic_ids(item_features)` — extract residuals `[L, 32]` and SID assignments.
2. Compute: L2 norm of `residuals[l]` (per item), entropy `H(c_l | c_{<l})` from empirical counts in training set, effective codebook utilization (count unique assigned codes at level l).
3. Save `results/residual_entropy_{dataset}.parquet`.
4. Compute Spearman ρ between per-dataset learned α_l and normalized residual entropy at level l.

**Files created:** `modules/analysis/residual_entropy.py`, `sagemaker/residual_entropy_estimator.py`
**Tests:** `tests/analysis/test_residual_entropy.py` — verify on a tiny 3-item synthetic RQ-VAE that norms and entropies are computed correctly.
**Effort:** 3h
**Dependencies:** Trained MTL decoders (PR 9)

### Task 5.8 — Integration test for the full stack

**What:** `tests/integration/test_mtl_e2e.py` — train an MTL decoder for 100 steps on a 200-item synthetic dataset, run inference with LevelAwareHybridDecoding + DBS base strategy, assert no crash and all beams valid.

**Effort:** 2h
**Dependencies:** Tasks 5.2, 5.4

---

## 6. Dataset Tasks

### Task 6.1 — Confirm existing Amazon loaders

**What:** Verify Beauty/Sports/Toys load correctly with `dataset_split` param; confirm item counts and sequence lengths match TIGER paper Table 1.
**Files modified:** None if correct; `data/amazon.py` if discrepancies found.
**Effort:** 1h

### Task 6.2 — Steam data loader

**What:** `data/steam.py` — Steam dataset loader following LIGER's preprocessing conventions exactly.

Steps:
1. Clone `facebookresearch/liger` to sibling directory `../liger/`.
2. Read `liger/data/steam_data.py` (or equivalent) — replicate: user/item filtering thresholds, sequence length cutoffs, train/val/test split ratios, negative sampling strategy.
3. Implement `RawSteam` class in `data/steam.py` mirroring `AmazonReviews` API.
4. Add `RecDataset.STEAM` enum entry in `data/processed.py`.
5. Document exact preprocessing in `data/steam_preprocessing.md` with item count, user count, and sequence stats matching LIGER's reported dataset statistics.

**Files created:** `data/steam.py`, `data/steam_preprocessing.md`
**Files modified:** `data/processed.py` (add STEAM enum entry)
**Tests:** `tests/data/test_steam_loader.py` — assert item count, user count, train/val/test sizes match LIGER's reported stats (from LIGER paper Table 1).
**Effort:** 4h
**Dependencies:** LIGER cloned in sibling directory

### Task 6.3 — Train missing RQ-VAE checkpoints

**What:** Launch SageMaker jobs to train missing RQ-VAE checkpoints. These are **blocking** for any downstream training.

| Job | Config | Instance | Estimated duration |
|---|---|---|---|
| Toys RQ-VAE | Clone of `configs/rqvae_amazon.gin` with `dataset_split="toys"` | `ml.g5.xlarge` spot | 4-6h |
| Steam RQ-VAE | New `configs/rqvae_steam.gin` | `ml.g5.xlarge` spot | 6-8h |

Save to `trained_models/rqvae_amazon_toys/` and `trained_models/rqvae_steam/`.

**Files created:** `configs/rqvae_steam.gin`
**Effort:** 1h to launch; wait for job completion
**Dependencies:** Task 6.2 (Steam data), Task 7 (SageMaker)

---

## 7. SageMaker Infrastructure Tasks

### Task 7.1 — Training Docker image

**What:** `docker/Dockerfile.training` — extends AWS SageMaker PyTorch DLC, installs `requirements.txt` + new `pyproject.toml` deps, copies source.

**Files created:** `docker/Dockerfile.training`
**Effort:** 2h

### Task 7.2 — Inference Docker image

**What:** `docker/Dockerfile.inference` — lighter image (no SageMaker training SDK), for eval jobs, alpha grid search, residual entropy.

**Files created:** `docker/Dockerfile.inference`
**Effort:** 1h

### Task 7.4 — SageMaker estimator scripts

**What:** `sagemaker/` directory with estimator wrappers and launch scripts.

```
sagemaker/
├── train_rqvae_estimator.py
├── train_decoder_estimator.py
├── train_decoder_mtl_estimator.py
├── eval_estimator.py
├── alpha_search_estimator.py
├── residual_entropy_estimator.py
└── launch/
    ├── launch_rqvae.py
    ├── launch_baselines.py
    ├── launch_mtl.py
    ├── launch_decoding_eval.py
    ├── launch_alpha_search.py
    └── launch_residual_entropy.py
```

S3 bucket: `s3://YOUR_S3_BUCKET/rqvae-level-aware/` (YOUR_AWS_PROFILE AWS profile). Jobs are launched from the compute machine, not this development machine.

All jobs use:
- `use_spot_instances=True`, `max_wait = 2 × max_run`
- Checkpoint to S3 every epoch
- Tags: `{"project": "rqvae-level-aware", "owner": "huseyin"}`
- Output: `s3://<bucket>/rqvae-level-aware/<experiment>/<job_name>/`

Instance types:
- RQ-VAE training: `ml.g5.xlarge`
- Decoder training: `ml.g5.2xlarge` (escalate to `ml.g5.4xlarge` for Steam if OOM)
- LIGER baseline: `ml.g5.2xlarge`
- Eval/alpha search/residual: `ml.g5.xlarge` spot

**Files created:** All above
**Effort:** 6h
**Dependencies:** Tasks 7.1–7.3

### Task 7.5 — Data upload script

**What:** `sagemaker/setup/upload_datasets.py` — runs existing data downloaders + Steam preprocessing, uploads processed files to `s3://<bucket>/rqvae-level-aware/data/<dataset>/`.

**Files created:** `sagemaker/setup/upload_datasets.py`
**Effort:** 1h
**Dependencies:** Task 6.2

### Task 7.6 — Result collection pipeline

**What:** `scripts/collect_results.py` — pulls all eval result JSONs from S3, assembles `results/all_runs.parquet`. All tables and figures are generated from this single parquet.

Standardized eval JSON schema:
```json
{
  "job_name": "string",
  "dataset": "beauty|sports|toys|steam",
  "decoder_type": "vanilla|mtl|liger",
  "decoding_strategy": "vanilla|dbs|gumbel|hybrid|level_aware_mix|sasrec_rerank",
  "alpha_schedule": [0.0, 0.0, 0.0],
  "seed": 42,
  "per_user_metrics": {"user_id": {"recall@5": 0.1, ...}},
  "aggregate": {"recall@5": 0.12, "ndcg@10": 0.08, ...}
}
```

**Files created:** `scripts/collect_results.py`
**Effort:** 2h

---

## 8. LIGER Baseline Plan

**Decision: Use LIGER paper results directly.** We cite Yang et al. 2024 Table 1 numbers rather than running LIGER's codebase. Standard RecSys practice when evaluation protocols match. Paper footnote: *"LIGER results cited from the original paper; our preprocessing follows LIGER's reported protocol."*

This eliminates: `docker/Dockerfile.liger`, all LIGER estimator/launch scripts, `scripts/convert_liger_results.py`, and the PR 5.5 gate.

### Task 8.1 — Record LIGER paper numbers

**What:** Create `results/baselines/liger_paper_numbers.json` with LIGER's published Recall@{5,10,20} and NDCG@{5,10,20} per dataset (Beauty, Sports, Toys, Steam) from LIGER Table 1.
**Why:** Single source of truth for LIGER numbers fed into `make_tables.py`.

**Files created:** `results/baselines/liger_paper_numbers.json`
**Effort:** 0.5h

### Task 8.2 — Steam stats verification

**What:** After implementing the Steam loader (Task 6.2), compare our processed dataset statistics (user count, item count, interaction count) against LIGER's reported Steam statistics. If they differ by >5%, add a footnote in the paper and do not make direct head-to-head Steam claims.
**Files modified:** `data/steam_preprocessing.md` (add comparison table)
**Effort:** 0.5h (after Steam loader exists)

### Task 8.3 — Statistical comparison methodology

Since we only have LIGER's aggregate means (not per-user arrays), comparison vs. LIGER uses: two-sample t-test with our per-user distribution against LIGER's reported mean, assuming LIGER's std from paper. Flag this as a weaker statistical comparison than the paired bootstrap we use for our own ablations.

---

## 9. Evaluation Matrix and Statistical Methodology

### Task 9.1 — Per-user metric infrastructure (PR 2)

**What:** Extend `evaluate/metrics.py` to support per-user tracking and add NDCG@K.

Changes:
- `TopKAccumulator.accumulate()` accepts an optional `user_ids: List[int]` argument.
- Stores `per_user: Dict[int, Dict[str, float]]` — user_id → {`recall@k`, `ndcg@k`}.
- `reduce()` returns both aggregate means and the per-user dict.
- Add `ndcg_at_k(ranked_list, target, k)` function.

**Files modified:** `evaluate/metrics.py`
**Files created:** None (add NDCG inline)
**Tests:** `tests/evaluate/test_metrics.py` — per-user tracking, NDCG at K=1/5/10/20.
**Effort:** 2h

### Task 9.2 — Paired bootstrap and effect sizes

**What:** `evaluate/stats.py` — statistical testing module.

```python
def paired_bootstrap_test(
    user_metrics_a: Dict[int, float],  # {user_id: metric_value} for config A
    user_metrics_b: Dict[int, float],  # config B
    n_resamples: int = 10000,
    alpha: float = 0.05,
) -> BootstrapResult:
    # Returns: mean_diff, ci_low, ci_high, p_value, cohens_d

def holm_bonferroni_correction(
    p_values: List[float],
    alpha: float = 0.05,
) -> List[bool]:  # significant[i] after correction
```

**Files created:** `evaluate/stats.py`
**Tests:** `tests/evaluate/test_stats.py` — verify against scipy bootstrap on small examples.
**Effort:** 2h

### Task 9.3 — Parquet result store

**What:** `evaluate/result_store.py` — writes standardized eval JSON (schema in §7.6) and reads `results/all_runs.parquet`.
**Files created:** `evaluate/result_store.py`
**Effort:** 1h

### Task 9.4 — Evaluation matrix (9 configs × 4 datasets × 5 seeds = 180 runs)

| ID | Decoder | Decoding | Alpha |
|---|---|---|---|
| B0 | vanilla | vanilla beam search | — |
| B1 | LIGER | LIGER's own | — |
| H1a | vanilla | codebook-aware DBS | — |
| H1b | vanilla | Gumbel top-k | — |
| H1c | vanilla | hybrid (det + stoch) | — |
| H1-best | vanilla | best of H1a/b/c per dataset | — |
| H2a | MTL | H1-best + SASRec rerank | swept α ∈ [0,1] |
| H2b-grid | MTL | H1-best + LevelAware | grid α, Optuna refined |
| **H2b-learned** | **MTL** | **H1-best + LevelAware** | **learned α_params** |

Beam size: k=50 for main table. Sweep k ∈ {5, 10, 20, 50, 100} for appendix.
Note: Increase `n_cands` from current 64 to `min(200, codebook_size)` for the main experiments — 64/256 sampling coverage is low.

### Task 9.5 — Per-segment slicing

Slice results by:
- User history quartiles (Q1=sparse, Q4=dense).
- Item popularity quartiles.
- Cold-start items (seen <5 times in training) — mandatory for LIGER comparison.

**Files modified:** `evaluate/metrics.py` (accept item popularity dict and mark cold-start items).
**Effort:** 2h

---

## 10. Paper Deliverables

### Task 10.1 — LaTeX directory structure

**What:** Create `paper/` with CIKM 2026 ACM format (9-page + refs).

```
paper/
├── paper.tex
├── acmart.cls
├── references.bib
├── sections/
│   ├── intro.tex
│   ├── related_work.tex
│   ├── method.tex
│   ├── experiments.tex
│   ├── analysis.tex
│   └── conclusion.tex
└── figures/        ← generated by scripts/make_figures.py
```

**Files created:** All above (paper.tex as skeleton with section includes)
**Effort:** 2h (skeleton only; writing happens continuously)

### Task 10.2 — Figure generation scripts

**What:** `scripts/make_figures.py` — reads `results/all_runs.parquet` and `results/residual_entropy_*.parquet`, generates all figures as PDF+PNG.

| Figure | Description | Source |
|---|---|---|
| F1 | TikZ schematic of per-level mixing during beam expansion | Hand-drawn (not auto-generated) |
| F2 | Main results bar chart — Recall@10 across configs × datasets with 95% CIs | `all_runs.parquet` |
| F3 | Residual norm and entropy vs. codebook level | `residual_entropy_*.parquet` |
| F4 | Alpha schedule (learned + grid) vs. level per dataset | `all_runs.parquet` alpha_schedule field |
| F5 | Scatter: learned α_l vs. normalized residual entropy | join of F3+F4 data |
| F6 | Per-segment slicing (sparse/dense users, head/tail items, cold-start) | `all_runs.parquet` with segment labels |
| F7 (appendix) | Beam size sweep | `all_runs.parquet` beam_size field |

**Files created:** `scripts/make_figures.py`
**Effort:** 4h (after results are available)

### Task 10.3 — Table generation scripts

**What:** `scripts/make_tables.py` — generates LaTeX table code from parquet.

Tables: T1 (main results with paired bootstrap p-values), T2 (ablations), T3 (per-segment on Beauty).
**Files created:** `scripts/make_tables.py`
**Effort:** 3h

---

## 11. PR Sequencing and Gating Reviews

### PR 0 — Repo setup (Tasks 3.1–3.6)
- Branches, CI, pre-commit, pyproject.toml, CONTRIBUTING.md, README notice.
- **Gate:** CI must pass on `develop` before anything merges.

### PR 1 — Codebase recon report
- `docs/codebase_findings.md` answering §2.1 and §2.2, including LIGER audit.
- No code changes. **Gating review: Huseyin approves before any implementation starts.**
- **[This plan document is PR 1.]**

### PR 2 — Eval harness (Tasks 9.1–9.3) — branch: `feat/eval-harness`
- Per-user Recall@K and NDCG@K, paired bootstrap, parquet store.
- **Gate:** All eval tests pass. Per-user tracking works correctly. NDCG values verified against a known example.

### PR 3 — Decoding framework refactor (Tasks 4.1–4.3) — branch: `feat/decoding-strategies`
- Extract vanilla beam search into new interface.
- **Gate:** `tests/decoding/test_vanilla_regression.py` passes with byte-for-byte identical output. This test is non-negotiable. If refactoring breaks parity, stop and fix before proceeding.

### PR 4 — SageMaker scaffolding (Tasks 7.1–7.5) — branch: `feat/sagemaker-infra`
- Three containers, all estimator scripts, data upload including Steam preprocessing.
- **Gate:** One end-to-end smoke test per dataset (100-step training + eval) completes without error on SageMaker.

### PR 5 — Vanilla TIGER baseline runs
- Train 4 vanilla decoders (Beauty, Sports, Toys, Steam) on SageMaker with 5 seeds each.
- **Gating review:** Record noise floor (mean ± std Recall@10 per dataset across 5 seeds). If std > 3pp, investigate before continuing — high noise may require more seeds for statistical power.
- **Checkpoint this gate:** Write results to `results/vanilla_baseline_review.md`.

### PR 5.5 — Record LIGER paper numbers (Task 8.1) — no branch needed
- Create `results/baselines/liger_paper_numbers.json` with LIGER Table 1 values.
- Verify Steam dataset stats match LIGER's reported statistics (Task 8.2).
- **Gate:** JSON file committed; Steam discrepancy (if any) documented.

### PR 6 — Codebook-aware DBS (Task 4.4)
- **Gate:** `test_dbs.py::test_diversity_monotone_in_lambda` passes; `test_trie_mask_compose.py` passes for DBS.

### PR 7 — Gumbel top-k (Task 4.5)
- **Gate:** `test_gumbel.py::test_unbiased_at_tau_one` (KL < 0.01); seed reproducibility test passes.

### PR 8 — Hybrid strategy (Task 4.6) + Pick H1-best per dataset
- Run H1a, H1b, H1c on all 4 datasets × 5 seeds. Pick H1-best per dataset.
- **Gating review:** Record H1 results. Identify whether codebook-aware DBS, Gumbel, or hybrid wins per dataset. This determines the base strategy for all H2 experiments.

### PR 9 — SASRec head + MTL training (Tasks 5.1–5.3)
- Train 4 MTL decoders on SageMaker.
- **Gating review:** SID metrics for MTL decoder must be within ±1% of vanilla baseline. If SID loss is >1% worse: reduce `λ_aux_max` or extend warmup. Document adjustment.

### PR 10 — SASRec reranker (Task 5.5)
- Run H2a (post-hoc reranking) with swept α across 4 datasets × 5 seeds.
- **CRITICAL gating review:** Compare H2a best vs. vanilla TIGER and vs. LIGER. This answers: "how much of the potential gain does simple post-hoc fusion capture?" If H2a ≥ LIGER on 3/4 datasets, the remaining contribution story is: "collaborative-signal head (our SASRec-style head) beats content-signal head (LIGER's Sentence-T5) even with the same post-hoc reranking mechanism." If H2a < LIGER: per-level mixing is needed to beat LIGER. In either case, **document the interpretation in a PR comment before proceeding.**

### PR 11 — Level-aware hybrid decoding, grid α (Tasks 5.4, 5.6)
- H2b-grid: coarse grid + Optuna refinement.
- **Gating review:** Does per-level α beat uniform α? Does the optimal α schedule tend to be monotone non-increasing? Document falsification/confirmation of the prior.

### PR 12 — Learned α variant (Task 5.6 learned)
- H2b-learned: freeze decoder, train alpha_params 1 epoch.
- **Gate:** `test_alpha_zero_equiv.py` passes for learned α.

### PR 13 — Residual entropy analysis (Task 5.7)
- **CRITICAL gating review:** Does Spearman ρ between learned α_l and residual entropy exceed 0.8? If yes: this is the paper's theoretical backbone — emphasize in intro and analysis. If no: document and analyze why; still include figure F3 and F5 with the actual correlation.

### PR 14 — Ablation suite
- Score normalization (z-score vs. temperature), per-level vs. uniform α, SASRec vs. text-embedding head inside our framework, base strategy composability.
- All on Beauty (minimum) + Sports.

### PR 15 — Per-segment slicing (Task 9.5)
- Cold-start analysis mandatory. Compare cold-start performance vs. LIGER.

### PR 16 — Paper draft (Tasks 10.1–10.3)
- Auto-generated tables and figures from parquet. LaTeX draft complete.
- Internal review before CIKM submission (May 23 deadline).

---

## 12. Risk Register

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| **Steam preprocessing diverges from LIGER's** | Medium | Medium | Compare dataset stats (users/items/interactions) against LIGER Table 1; if >5% difference, footnote in paper and don't claim direct Steam comparison. |
| **LIGER paper evaluation protocol differs from ours** | Low | Medium | We use the same leave-one-out + Recall@K + NDCG@K protocol that TIGER/LIGER share; footnote any known differences. |
| **Monotone α prior falsified** | Medium | Low | Falsified prior + explanation is paper-worthy. Still publish with the actual measured schedule and residual entropy correlation. |
| **Per-level mixing does not beat SASRec reranker** | Medium | Medium | Pivot: "collaborative-signal head beats content-signal head; per-level analysis explains when in-loop mixing helps." Still publishable because the head-architecture comparison is novel. |
| **SASRec aux loss destabilizes SID training** | Low | High | λ_aux warmup, grad clip norm 1.0, monitor SID loss curve; abort if >1% degradation vs. vanilla. |
| **Score scale mismatch (α meaningless)** | Low | High | Z-score normalization default. Unit test: α=0 must exactly reproduce base strategy. |
| **Beam validity broken by noise injection** | Low | High | `test_trie_mask_compose.py` enforces validity for every strategy. |
| **Spot instance interruption mid-training** | Medium | Low | Checkpoint every epoch; max_wait=2×max_run. |
| **Beauty noise floor swamps effect size** | Low | Medium | 5 seeds, paired bootstrap, report all 4 datasets. |
| **RQ-VAE checkpoints stale/incompatible** | Low | High | PR 1 verification: load each checkpoint and run `get_semantic_ids` on a batch; assert no error and check output shapes before launching decoder training. |
| **CIKM deadline slips** | Low | Medium | Parallel submission to RecSys 2026 Industry track (May 21) if Amazon Music production results available. |
| **n_cands=64 sampling coverage too low** | Medium | Medium | Increase to `min(200, vocab_size)` for main experiments. Report sensitivity in appendix. |
| **Codebook dim 32 << decoder dim 384 alignment** | Low | Medium | SASRec head must project to 32-d (not 384-d) for per-level mixing. See §1.8. |

---

## 13. Decisions Log

All questions from the original plan have been resolved:

1. **LIGER reproduction:** Use paper results directly. Cite LIGER Table 1 with footnote.
2. **S3 bucket:** `YOUR_S3_BUCKET` under YOUR_AWS_PROFILE AWS profile. SageMaker infrastructure targets this bucket. Development machine only; jobs will be launched from the compute machine.
3. **LIGER clone location:** `../liger/` if ever needed for reference; not needed for experiments.
4. **Beauty RQ-VAE checkpoint:** Use `checkpoint_399999.pt` (matches existing `decoder_amazon.gin`).
5. **n_cands:** Increase to `min(200, 256)` for all main experiments. Run sensitivity sweep at n_cands ∈ {64, 128, 200} in appendix figure F7.
6. **Pre-existing decoder checkpoint:** Ignore — train fresh decoders for all 4 datasets.
7. **RecSys 2026 Industry track:** Not pursuing.
8. **Token-ID DBS ablation:** Yes — include one T2 row with token-ID-based DBS (matching Penha et al. 2024) vs. our codebook-embedding DBS.
9. **NDCG computation:** Binary-relevance NDCG with single positive (true next item). `NDCG@K = 1/log2(r+1)` if rank r ≤ K, else 0. Averaged over all users. IDCG=1 so no additional normalization.
10. **Steam dataset source:** McAuley Lab, `https://cseweb.ucsd.edu/~jmcauley/datasets/steam/`. 5-core filtering (≥5 interactions per user and item), leave-one-out split. Expected ~334K users, ~13K items, ~3M interactions.

---

## Appendix: File Creation Summary

### New directories
```
modules/decoding/
modules/heads/
modules/analysis/
evaluate/
sagemaker/
sagemaker/launch/
sagemaker/setup/
docker/
data/ (add steam.py)
scripts/
tests/decoding/
tests/heads/
tests/training/
tests/evaluate/
tests/analysis/
tests/integration/
paper/
paper/sections/
paper/figures/
results/
docs/
envs/
configs/ (new gin files)
```

### New files (key)
```
modules/decoding/__init__.py, base.py, vanilla.py, dbs.py, gumbel.py, hybrid.py,
    level_aware_mix.py, sasrec_reranker.py
modules/heads/__init__.py, sasrec_head.py
modules/analysis/residual_entropy.py
evaluate/stats.py, result_store.py
data/steam.py, steam_preprocessing.md
train_decoder_mtl.py
configs/decoder_amazon_beauty_mtl.gin, decoder_amazon_sports_mtl.gin,
    decoder_amazon_toys_mtl.gin, decoder_steam_mtl.gin, rqvae_steam.gin
sagemaker/{all estimators and launch scripts}
docker/Dockerfile.{training,inference,liger}
scripts/alpha_grid_search.py, train_alpha_params.py, collect_results.py,
    make_figures.py, make_tables.py
tests/{all test files listed per section}
paper/paper.tex, sections/*.tex, references.bib
pyproject.toml, .github/workflows/ci.yml, .pre-commit-config.yaml, CONTRIBUTING.md
docs/codebase_findings.md, liger_compatibility.md
envs/liger_environment.yml
```

### Modified files (key)
```
modules/model.py          — thin generate() loop calling strategy.expand()
evaluate/metrics.py       — per-user tracking, NDCG@K
train_decoder.py          — extract _train_loop() shared helper
data/processed.py         — add RecDataset.STEAM
README.md                 — research fork notice
LICENSE                   — derivative work notice
requirements.txt          — keep; pyproject.toml adds new deps
```
