# RQ-VAE collapse bisect runbook

Use this on a machine that has the AWS profile for the training account (the
one whose SageMaker execution role you've been using), `mwinit` if you need
Midway, and `git`. The three experiments below run as short SageMaker jobs
(≤ 90 min each on `ml.g5.4xlarge`), so you can launch them in parallel and
compare verdicts in S3.

## Why this exists

All four `repro3` full trainings collapsed by step ~12k despite both earlier
fixes being correctly reverted (`57808c5` in `f9b645a`, `0449747` in
`4e0eb00` + `baa1190`). The single non-reverted RQ-VAE-training-path commit
between the healthy-era Beauty ckpt and HEAD is `a5367ed` (SiLU + decoder
L2-norm), and its "match Beauty arch" justification cannot be verified from
the saved `model_config` because neither field is an `RqVae.__init__`
kwarg at that era. The other two open hypotheses are a `@torch.compile` +
ROTATION_TRICK interaction and commitment-loss dominance once the encoder
approaches the codebook. See `AGENTS.md` → "Open questions".

Each experiment below is a single-variable 5k-step sanity run, bundled with
the validator so the verdict comes back as JSON alongside the SageMaker job.

## 0. Prereqs (one-time)

```bash
# Anywhere you want the repo. Replace $YOUR_AWS_PROFILE, $YOUR_S3_BUCKET, and
# $YOUR_SM_EXECUTION_ROLE below with the values for your account.
git clone <this-repo> rqvae-bisect && cd rqvae-bisect
git checkout develop
git pull

# Python 3.11 preferred (matches the SageMaker container the jobs run on).
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install 'sagemaker>=2.230,<3' 'boto3>=1.34'
# No need to install torch / gin / einops locally — they only run inside the
# SageMaker container.

# Auth (whatever your org requires). Examples:
mwinit              # Midway-protected accounts
aws sso login --profile $YOUR_AWS_PROFILE
```

If your AWS profile / region / S3 bucket / execution-role ARN differ from the
hard-coded defaults in `sagemaker/launch/launch_rqvae*.py`, either edit those
constants once at the top of the file or export the overrides into your shell
before launching — the launchers already read `boto3.Session(profile_name=...)`
so a profile switch is enough for auth.

## 1. Bisect experiment A — revert `a5367ed` (activation + decoder norm)

Hypothesis: `a5367ed` switched encoder/decoder MLP from ReLU to SiLU **and**
turned the decoder's trailing L2-norm on. Neither is in the saved Beauty
`model_config` (they weren't `__init__` kwargs at that era), so the "match
Beauty arch" premise cannot be verified from the healthy checkpoint. If this
experiment validates HEALTHY at 5k steps while the baseline below collapses,
`a5367ed` is the regression.

```bash
# A1. Baseline — current HEAD (SiLU + decoder.normalize=True, a5367ed applied).
python sagemaker/launch/launch_rqvae_sanity.py \
    --datasets beauty \
    --iterations 5000 \
    --job-suffix bisect-baseline

# A2. Revert a5367ed — ReLU + decoder.normalize=False.
python sagemaker/launch/launch_rqvae_sanity.py \
    --datasets beauty \
    --iterations 5000 \
    --gin-config configs/rqvae_amazon_beauty_bisect_a5367ed.gin \
    --job-suffix bisect-a5367ed
```

Verdicts:

```
s3://$YOUR_S3_BUCKET/rqvae-sanity/beauty-bisect-baseline/.../output/model.tar.gz
s3://$YOUR_S3_BUCKET/rqvae-sanity/beauty-bisect-a5367ed/.../output/model.tar.gz
```

Inside each `model.tar.gz` is a `verdict.json` with per-level `unique_sids`,
`entropy_bits`, and `min_codebook_distance`. A `healthy: true` at 5k is
necessary but not sufficient — also compare W&B `rqvae_loss` curves and the
new `residual_avg_norm_{0,1,2}` / `emb_avg_norm_{i}` series. If the
a5367ed-reverted run keeps `vl` non-zero across the full 5k while the
baseline's `vl` is already dropping, that's your answer.

## 2. Bisect experiment B — disable `torch.compile`

Hypothesis: some interaction between `@torch.compile(mode="reduce-overhead")`,
`accelerate`, and `ROTATION_TRICK` gradient flow regressed between the
healthy era and now. The `kmeans_initted` buffer fiasco (`0449747` →
`4e0eb00`) showed that `torch.compile` can silently trap Python-level
invariants; this tests whether another such trap is still present.

```bash
python sagemaker/launch/launch_rqvae_sanity.py \
    --datasets beauty \
    --iterations 5000 \
    --disable-compile \
    --job-suffix bisect-nocompile
```

`--disable-compile` passes `RQVAE_DISABLE_COMPILE=1` to the container; the
`_maybe_compile` module-level helper in `modules/rqvae.py` turns the decorator
into an identity when that env var is set. Compare this verdict and W&B
curves against the `bisect-baseline` run.

If Beauty collapses at 5k with compile on and stays healthy with compile off,
promote this to a full 400k run:

```bash
python sagemaker/launch/launch_rqvae.py --dataset beauty \
    --disable-compile --job-suffix nocompile
```

## 3. Bisect experiment C — `commitment_weight=0.1`

Safety-net hypothesis: the `vl` oscillation down to 0 looks like commitment
loss overwhelming reconstruction once the encoder is close enough to the
codebook. Lower the weight to give the optimizer more room.

```bash
python sagemaker/launch/launch_rqvae_sanity.py \
    --datasets beauty \
    --iterations 5000 \
    --gin-config configs/rqvae_amazon_beauty_bisect_cw01.gin \
    --job-suffix bisect-cw01
```

## 4. Inspecting the verdicts

The validator writes `verdict.json` to `/opt/ml/model/verdict.json`, which
SageMaker packs into `output/model.tar.gz`. Pull it back locally:

```bash
aws s3 cp \
    s3://$YOUR_S3_BUCKET/rqvae-sanity/beauty-bisect-a5367ed/<jobname>/output/model.tar.gz \
    /tmp/bisect-a5367ed.tar.gz \
    --profile $YOUR_AWS_PROFILE
tar -xzf /tmp/bisect-a5367ed.tar.gz -C /tmp/ verdict.json
cat /tmp/verdict.json
```

`healthy: true` means every level has ≥ 10 unique SIDs **and**
`min_codebook_distance ≥ 0.01` **and** level-0 entropy > 2 bits
(`evaluate/validate_rqvae.py:222`).

## 5. W&B curves to watch

Old metrics (unchanged semantics):

- `rqvae_loss` — the VQ + commitment loss. Collapses to 0 in the broken runs.
- `emb_avg_norm_{0,1,2}` — per-level codebook-vector norms.
- `p_unique_ids` — fraction of distinct SIDs in the batch (post-`torch.compile`
  scalar; not a full-codebook histogram — use the validator for that).

New metrics (added for this bisect):

- `residual_avg_norm_{0,1,2}` — per-level pre-quantization residual `||x||`.
  - If this **shrinks** before `vl` hits 0 → rotation-trick scale collapse.
  - If this stays stable but `emb_avg_norm_i` collapses → codebook dying to a
    point, commitment-loss dominance is plausible.
  - If both stay stable but `vl` still goes to 0 → look at the gradient path
    (rotation math), not loss balance.

## 6. If everything collapses

If all four 5k runs come back collapsed, the issue is earlier than `a5367ed`
and not in the three hypothesised mechanisms. At that point:

1. Re-check input data: hash-compare `s3://.../datasets/amazon/` against what
   the healthy Beauty ckpt read (the cache may have been regenerated with
   different Sentence-T5 weights or preprocessing).
2. Try `torch==2.4.x` in the training container (current is 2.5.1). Rotation
   trick autograd has seen fixes across 2.3→2.5.
3. Rebuild the healthy Beauty checkpoint's environment commit-by-commit by
   checking out whatever `15126b7` was close to (the exact SHA isn't in
   `origin/develop`; `2bfce87` at 2026-04-17 is the closest walkable
   left-anchor on this repo).

## 7. Launching other datasets once Beauty is healthy

`launch_rqvae_sanity.py` takes `--datasets beauty sports toys steam` in any
combination. `launch_rqvae.py` is the full 400k-step launcher and accepts the
same `--disable-compile` / `--gin-config` / `--job-suffix` overrides. Don't
kick off cross-dataset full trainings until Beauty is verifiably healthy
again — that's the fastest way to ration the g5.4xlarge quota against a
known-good reference.
