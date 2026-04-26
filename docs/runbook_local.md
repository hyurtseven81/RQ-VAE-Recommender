# Local runbook — laptop-side steps for the CIKM 2026 pipeline

You are an automation agent. Follow this runbook end-to-end on the
operator's laptop. The repo is already cloned and pulled to the current
working directory; do not re-clone.

This runbook covers everything that happens **off** SageMaker:

- Stage 0 — the inventory + ML32M pipeline gate.
- Post-SageMaker — pulling result tarballs from S3, building the
  paper's `all_runs.parquet`, and rendering tables/figures.

Every long-running cloud step is in `docs/runbook_sagemaker.md`. Don't
launch SageMaker jobs from this runbook.

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
pip install --quiet \
    gin-config==0.5.0 einops 'polars==1.9.0' \
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

This is the cheap pre-flight before spending SageMaker quota. It
checks (a) checkpoints load, (b) `ItemData` constructs for each
dataset, and writes `docs/paper_plan_stage0_report.md`.

### 1.1 Run the gate

```bash
set -a; source .env; set +a
PYTHONPATH=. python scripts/stage0_pipeline_check.py \
    --datasets beauty sports ml32m \
    --report-path docs/paper_plan_stage0_report.md \
    --json-path /tmp/stage0_report.json
```

The script exits 0 iff every requested dataset passes both
`checkpoint_loads` and (unless `--skip-data-load` was passed)
`data_loads`. Show the report:

```bash
cat docs/paper_plan_stage0_report.md
```

### 1.2 Decision gate

Inspect the report's verdict column and act:

- **All three READY** → `§1: PASS`. The SageMaker runbook is unblocked
  for all three datasets.
- **Beauty + Sports READY, ML32M BLOCKED on `data_loads`** → record
  the failure mode in `docs/progress_log.md`. Status: `§1: PARTIAL —
  ML32M dropped pending data-pipeline fix`. The SageMaker runbook
  proceeds with `--datasets beauty sports` only.
- **ML32M BLOCKED on `checkpoint_loads`** → either the upstream sync
  in §0.4 didn't pull the file, or the gin config's architecture
  doesn't match the saved `model_config`. Stop and ask the operator;
  do not attempt to fix the gin config without explicit instruction.
- **Beauty or Sports BLOCKED** → that's a regression, not a config
  issue. Stop and report the full traceback line; do not proceed.

### 1.3 (Optional) Local validator runs

If you want a per-checkpoint codebook fingerprint without spending
SageMaker time, the validator can run on CPU locally. This takes
~5–15 min per dataset depending on item count. Skip if SageMaker
validator jobs are already queued (`docs/runbook_sagemaker.md` §1).

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

## §2 Sync preprocessed dataset caches (only if Stage 1.3 in the SageMaker runbook will run locally)

The SageMaker training and eval jobs mount preprocessed dataset caches
from S3, so this step is **not required** to launch SageMaker runs.
Skip §2 entirely if you are only orchestrating SageMaker.

```bash
mkdir -p dataset
aws s3 sync "$RQVAE_S3_BASE/datasets/amazon/" dataset/amazon/ \
    --profile "$RQVAE_AWS_PROFILE" --exact-timestamps --no-progress
aws s3 sync "$RQVAE_S3_BASE/datasets/ml-32m/" dataset/ml-32m/ \
    --profile "$RQVAE_AWS_PROFILE" --exact-timestamps --no-progress
ls dataset/amazon/ dataset/ml-32m/ 2>/dev/null
```

If `dataset/ml-32m/` does not exist on S3 yet, this is expected — see
the SageMaker runbook for the preprocessing job, or skip ML32M for now.

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
    echo "## $(date -u +%FT%TZ) — local runbook session"
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
