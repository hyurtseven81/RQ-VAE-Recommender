# Operator runbook — pre-flight + post-run for the CIKM 2026 pipeline

You are an automation agent. Follow this runbook end-to-end on the
operator's machine. The repo is already cloned and pulled to the
current working directory; do not re-clone.

This runbook covers everything the operator does **around** the
SageMaker pipeline:

- Pre-flight — venv, AWS auth, env vars, upstream `trained_models/`
  sync, presence-only Stage 0 gate.
- Post-SageMaker — pulling result tarballs from S3, building the
  paper's `all_runs.parquet`, rendering tables and figures.

The actual training and evaluation jobs run on SageMaker via the
companion `docs/runbook_sagemaker.md`. The deep checkpoint-load /
codebook-fingerprint validation also runs on SageMaker (via the
validator job in §1 there) — this runbook does **not** import the
heavy ML stack and does **not** launch any SageMaker jobs.

## Conventions for the agent

- Run every command from the repo root unless stated otherwise.
- Start each shell invocation with `set -a; source .env; set +a` so
  `RQVAE_S3_BASE` and friends are loaded.
- If a step fails, stop and report the failing command + last 20 lines
  of output. Do **not** silently retry.
- At the end of each top-level section, post a one-line status
  (`SECTION X: PASS` or `SECTION X: FAIL: <one-sentence reason>`).
- At the end of the runbook, append a dated paragraph to
  `docs/progress_log.md` (create if it does not exist) summarising what
  ran, what passed, what failed, and what should run next.

## §0 Pre-flight

### 0.1 Env file

`.env` should already exist at the repo root with the four required
variables. Verify without echoing values:

```bash
test -f .env || { echo "MISSING: .env"; exit 1; }
awk -F= '/^[[:space:]]*RQVAE_(S3_BASE|SAGEMAKER_ROLE|AWS_PROFILE|AWS_REGION)[[:space:]]*=/ \
    {print $1 ": SET"}' .env
```

You should see four `: SET` lines. If any are missing, stop and ask the
operator to populate `.env` (template: `docs/runbook_sagemaker.md` §0).

### 0.2 Python venv

```bash
test -d .venv || python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip --quiet

# Launcher deps (always needed)
pip install --quiet 'sagemaker>=2.230,<3' 'boto3>=1.34'

# Stage 0 + local validator + collect_results need the ML stack:
pip install --quiet 'torch>=2.5.1' 'torchvision>=0.20.1' \
    --index-url https://download.pytorch.org/whl/cpu
# Note: protobuf is pinned to <6 because wandb 0.19's generated
# bindings are incompatible with protobuf 6+. Without this pin a
# stray transitive install (tf-keras etc.) can leave the venv with
# protobuf 7, which crashes any entry point that imports wandb —
# including train_rqvae, which Stage 0 imports to register gin
# configurables. See AGENTS.md "Known issues" + requirements.txt.
pip install --quiet \
    gin-config==0.5.0 einops 'polars==1.9.0' \
    'protobuf>=5.26.1,<6' \
    'wandb>=0.19.0,<0.25' \
    sentence-transformers accelerate pandas pyarrow \
    huggingface_hub tqdm
```

If `torch` is already installed but mismatched with `torchvision`
(symptom: `RuntimeError: operator torchvision::nms does not exist`),
force-reinstall both at matched versions:

```bash
pip install --upgrade --force-reinstall \
    'torch==2.5.1' 'torchvision==0.20.1' \
    --index-url https://download.pytorch.org/whl/cpu
```

If a previous session installed `tf-keras`, `tensorflow`, or anything
that pulls in protobuf 6+ (symptom on any Stage 0 / validator run:
`TypeError: Couldn't build proto file into descriptor pool: …` or
`AttributeError: …protobuf.internal.builder…`), force-downgrade
protobuf:

```bash
pip install --quiet --force-reinstall 'protobuf>=5.26.1,<6'
```

### 0.3 AWS auth

The operator's organisation determines the auth flow. Try one of:

```bash
mwinit                                          # Midway / Amazon
aws sso login --profile "$RQVAE_AWS_PROFILE"    # AWS SSO
```

Verify credentials are live by listing the project bucket prefix:

```bash
aws s3 ls "$RQVAE_S3_BASE/" --profile "$RQVAE_AWS_PROFILE" | head -5
```

A non-empty listing or an empty-but-no-error response is fine; an
`AccessDenied` / `ExpiredToken` means credentials are stale. Refresh
and retry once.

### 0.4 Sync upstream RQ-VAE checkpoints into trained_models/

`trained_models/` is gitignored locally and on SageMaker. The Stage 0
checks read these files; the SageMaker training jobs upload them with
`source_dir="."` so they need to be present locally even though the
fork doesn't track them.

```bash
git remote add upstream https://github.com/EdoardoBotta/RQ-VAE-Recommender.git 2>/dev/null || true
git fetch upstream main
# Pull just the trained_models/ tree, then unstage so it stays untracked.
git checkout upstream/main -- trained_models/
git reset HEAD trained_models/ 2>/dev/null || true
ls trained_models/rqvae_amazon_beauty trained_models/rqvae_amazon_sports trained_models/rqvae_ml32m
```

You should see at least one `.pt` file under each of the three
directories. If any directory is missing, stop and report.

Status: `§0: PASS` once all four sub-steps complete clean.

## §1 Stage 0 — local pipeline gate

This is the **cheap pre-flight** before spending SageMaker quota.
Default mode is presence-only — it just verifies that the upstream
RQ-VAE checkpoints exist locally with non-zero size. The deep
checkpoint-load validation (which requires the full ML stack and is
slow on laptop CPUs) is delegated to the SageMaker validator
(`docs/runbook_sagemaker.md` §1), which is strictly more thorough — it
also reports per-level codebook fingerprints.

> ⚠️ Past incidents:
>
> - A previous session left the data-load gate on with no local
>   cache, which silently triggered raw ML32M preprocessing (32M-rating
>   download + Sentence-T5 over ~86k items on CPU) and ran for 13+
>   hours. The script now refuses to materialise raw data and the
>   default-on `--skip-data-load` flag below double-protects against
>   a re-run of that.
> - A more recent session blocked on a venv `protobuf` mismatch that
>   broke wandb's import — every `_check_checkpoint_loads` call
>   crashed at module-import time. The default-on `--skip-checkpoint-load`
>   flag below side-steps that entirely; the SageMaker validator (which
>   uses the SM container's pinned env) handles the deep check.

### 1.1 Run the gate (presence-only, default cheap path)

```bash
set -a; source .env; set +a
PYTHONPATH=. python scripts/stage0_pipeline_check.py \
    --datasets beauty sports ml32m \
    --skip-checkpoint-load \
    --skip-data-load \
    --report-path docs/paper_plan_stage0_report.md \
    --json-path /tmp/stage0_report.json
```

This finishes in seconds and only checks `trained_models/<dataset>/
checkpoint_high_entropy.pt` exists with reasonable size. It does **not**
import torch / gin / wandb so it can't be derailed by venv issues.
Show the report:

```bash
cat docs/paper_plan_stage0_report.md
```

### 1.2 Decision gate

Inspect the report's `checkpoint_present` column:

- **All three present** → `§1: PASS`. Proceed to
  `docs/runbook_sagemaker.md`. The SageMaker validator there does the
  actual checkpoint-load + codebook-fingerprint check; that is the
  authoritative gate for the paper.
- **One or more missing** → §0.4 didn't pull that file. Re-run the
  upstream `git checkout upstream/main -- trained_models/` step and
  retry. If the upstream repo doesn't have a checkpoint for that
  dataset, drop it from this run and inform the operator.

### 1.3 (Optional) Deep checkpoint-load gate — only if you want a local sanity check

The SageMaker validator already does this on the cloud, so this is
strictly redundant. Run it locally only if you want to catch an
architecture mismatch before paying for a SageMaker job. It needs the
full ML stack from §0.2 installed and uncorrupted.

```bash
set -a; source .env; set +a
PYTHONPATH=. python scripts/stage0_pipeline_check.py \
    --datasets beauty sports ml32m \
    --skip-data-load \
    --report-path docs/paper_plan_stage0_report.md \
    --json-path /tmp/stage0_report.json
```

Decision tree for `checkpoint_loads`:

- **All three READY** → architecture matches, proceed with confidence.
- **All three BLOCKED on the same import-time error** (e.g. wandb /
  protobuf / sentence-transformers / torchvision) → this is a venv
  problem, not a codebase regression. Most common: a stray
  `tf-keras` / `tensorflow` install left protobuf at 7.x, which
  crashes wandb's import. Apply the venv-repair commands at the end
  of §0.2 and re-run, **or** just skip §1.3 and let the SageMaker
  validator do this check.
- **One dataset BLOCKED with a unique error** → real regression for
  that dataset. Stop and report the full traceback line.

### 1.4 (Optional) Local validator runs — only if you want a local fingerprint

If you want a per-checkpoint codebook fingerprint without spending
SageMaker time, the validator can run on CPU locally. This takes
~5–15 min per dataset depending on item count. Skip if SageMaker
validator jobs are already queued (`docs/runbook_sagemaker.md` §1).

> Pre-requirement: §2 must have synced the relevant `dataset/<split>/`
> cache. The validator will hard-fail with an actionable error if the
> processed cache is missing — it no longer silently triggers raw
> preprocessing.

```bash
mkdir -p /tmp/validate-local
for d in beauty sports ml32m; do
    case $d in
        beauty) ckpt=trained_models/rqvae_amazon_beauty/checkpoint_high_entropy.pt; gin=configs/rqvae_amazon_beauty.gin ;;
        sports) ckpt=trained_models/rqvae_amazon_sports/checkpoint_high_entropy.pt; gin=configs/rqvae_amazon_sports.gin ;;
        ml32m)  ckpt=trained_models/rqvae_ml32m/checkpoint_high_entropy.pt;          gin=configs/rqvae_ml32m.gin ;;
    esac
    PYTHONPATH=. python evaluate/validate_rqvae.py \
        --config_path "$gin" \
        --rqvae_checkpoint "$ckpt" \
        --output_dir "/tmp/validate-local/$d" || true
    cat "/tmp/validate-local/$d/verdict.json" 2>/dev/null || echo "$d: no verdict"
done
```

Record each verdict's per-level fingerprint in
`docs/paper_plan_stage0_report.md` under a "Local fingerprints"
section. Known reference: Beauty should be ~248–256/256 unique SIDs
with entropy 7.6–7.7 bits; Sports is partial-collapse (33/96/134, 1.78
bit L0).

Status: `§1: PASS` if Stage 0 gate is satisfied for at least Beauty + Sports.

## §2 Sync preprocessed dataset caches (optional)

The SageMaker training and eval jobs mount preprocessed dataset caches
from S3, so this step is **not required** to launch SageMaker runs.
Run §2 only if the operator wants to do one of these locally:

- The optional full data-load gate (§1.3 above).
- The optional local validator runs (§1.4 below).

Sync only the splits you need. The Amazon zip is small (~hundreds of
MB); ML32M can be tens of GB.

```bash
set -a; source .env; set +a
mkdir -p dataset

# Amazon (Beauty + Sports + Toys share the same cache):
aws s3 sync "$RQVAE_S3_BASE/datasets/amazon/" dataset/amazon/ \
    --profile "$RQVAE_AWS_PROFILE" --exact-timestamps --no-progress

# ML32M — only if the operator wants the local data-load gate to cover it:
# aws s3 sync "$RQVAE_S3_BASE/datasets/ml-32m/" dataset/ml-32m/ \
#     --profile "$RQVAE_AWS_PROFILE" --exact-timestamps --no-progress

ls dataset/amazon/processed/ dataset/ml-32m/processed/ 2>/dev/null
```

If a target prefix does not exist on S3 yet, the cache has never been
generated — that's a SageMaker-side preprocessing job
(`sagemaker/launch/launch_preprocess_datasets.py`), not a local one.
Drop the affected dataset from the local-side gates and let the
SageMaker runbook handle it.

> Do **not** run `ItemData(...)` directly against an empty
> `dataset/<split>/` directory locally — it will silently invoke
> `raw_data.process()` which downloads the source dataset and runs
> Sentence-T5 over every item on CPU. The Stage 0 script (§1.3) now
> guards against this; nothing else here does.

## §3 Post-SageMaker — pull results from S3

Run this after the operator confirms (via the SageMaker runbook) that
all eval jobs have status `Completed`. This step downloads result
tarballs from S3, unpacks them, and writes
`results/stage{1,2}/all_runs.parquet`.

### 3.1 Stage 1 aggregate

```bash
set -a; source .env; set +a
PYTHONPATH=. python scripts/collect_results.py \
    --prefix rqvae-level-aware/eval-results/baseline \
    --aggregate-output results/stage1/all_runs.parquet \
    --per-user-output  results/stage1/per_user_runs.parquet
```

Confirm the printed group-by reports the expected
`(dataset × decoder_type × strategy)` combinations. For Stage 1 you
expect 4 strategies × N datasets (where N is 2 or 3 depending on
ML32M's Stage-0 verdict): `vanilla`, `dbs`, `gumbel_topk`, `hybrid`.

### 3.2 Stage 2 aggregate

```bash
PYTHONPATH=. python scripts/collect_results.py \
    --prefix rqvae-level-aware/eval-results/mtl \
    --aggregate-output results/stage2/all_runs.parquet \
    --per-user-output  results/stage2/per_user_runs.parquet
```

Stage 2 should report `sasrec_rerank`, `level_aware_mix` (with the
refined-grid winner), `level_aware_mix_grid`, `level_aware_mix_learned`
per dataset.

### 3.3 (Optional) Stage 2 grid-search CSVs

These are auxiliary — they show every alpha point's metrics from the
3³ pilot and 5³ refined grids. Useful for the paper's heatmap figure.

```bash
mkdir -p results/stage2/alpha_search
aws s3 sync "$RQVAE_S3_BASE/alpha-search/" results/stage2/alpha_search/ \
    --profile "$RQVAE_AWS_PROFILE" --exact-timestamps --no-progress
# Each job's output.tar.gz is at .../<job>/output/output.tar.gz
find results/stage2/alpha_search/ -name 'output.tar.gz' -exec sh -c '
    d=$(dirname "$1")
    tar -xzf "$1" -C "$d"
' _ {} \;
ls results/stage2/alpha_search/*/*/output/*.csv 2>/dev/null | head -5
```

### 3.4 (Optional) Pull learned-α checkpoints

```bash
mkdir -p results/stage2/alpha_learned
aws s3 sync "$RQVAE_S3_BASE/alpha-learned/" results/stage2/alpha_learned/ \
    --profile "$RQVAE_AWS_PROFILE" --exact-timestamps --no-progress
find results/stage2/alpha_learned/ -name 'model.tar.gz' -exec sh -c '
    d=$(dirname "$1")
    tar -xzf "$1" -C "$d"
' _ {} \;
find results/stage2/alpha_learned/ -name '*_learned.summary.json' -exec cat {} \;
```

Status: `§3: PASS` once both `results/stage{1,2}/all_runs.parquet`
exist and the group-by output covers every expected
`(dataset, decoder_type, strategy)` combination.

## §4 Paper artefacts

```bash
mkdir -p paper/tables paper/figures
PYTHONPATH=. python scripts/make_tables.py \
    --results results/stage1/all_runs.parquet \
    --output paper/tables/ || true   # also retry combined below

# Stage 2 numbers feed the same table generator — concatenate first:
python -c "
import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd
s1 = pq.read_table('results/stage1/all_runs.parquet').to_pandas()
s2 = pq.read_table('results/stage2/all_runs.parquet').to_pandas()
combined = pd.concat([s1, s2], ignore_index=True)
pq.write_table(pa.Table.from_pandas(combined), 'results/all_runs.parquet')
print(f'Combined {len(combined)} rows -> results/all_runs.parquet')
"

PYTHONPATH=. python scripts/make_tables.py \
    --results results/all_runs.parquet \
    --output paper/tables/

PYTHONPATH=. python scripts/make_figures.py \
    --results results/all_runs.parquet \
    --output paper/figures/
```

Inspect the outputs:

```bash
ls -la paper/tables/ paper/figures/
```

If `make_tables.py` warns about a missing LIGER baseline, that's
expected — `results/baselines/liger_paper_numbers.json` holds
placeholder values per AGENTS.md and the table generator skips the B1
row gracefully.

Status: `§4: PASS` once `paper/tables/` has at least the main results
table file and `paper/figures/` has at least one figure (alpha
heatmap or strategy bar chart).

## §5 Final report

Append a dated paragraph to `docs/progress_log.md`:

```bash
{
    echo
    echo "## $(date -u +%FT%TZ) — operator runbook session"
    echo
    echo "Stage 0 verdict: <READY for {datasets} | BLOCKED on ml32m | …>"
    echo "Datasets in scope: <beauty / sports / ml32m | beauty / sports>"
    echo "Aggregated: results/stage1/all_runs.parquet (<N1> rows),"
    echo "            results/stage2/all_runs.parquet (<N2> rows),"
    echo "            results/all_runs.parquet (<N1+N2> rows)"
    echo "Paper artefacts: paper/tables/* and paper/figures/* generated."
    echo "Open issues: <one-liner | none>"
    echo "Next: <SageMaker runbook step / paper edit / nothing>"
} >> docs/progress_log.md
```

Replace `<…>` with the actual values from the run. Then post a
two-line summary to the operator: which sections passed, which (if
any) failed, and the next concrete step.
