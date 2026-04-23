# Agent runbook — CIKM 2026 experimental pipeline

This is the executable form of `docs/paper_plan.md`. An agent (Kiro, Claude
Code, etc.) can follow it start to finish. The paper plan is the "what"; this
file is the "how".

Every step below is idempotent — re-running it when nothing has changed is
safe. Every long-running step has a corresponding **monitor** step at the
end of this file that polls its progress.

Unless explicitly noted, all commands run from the repo root.

## 0. Operator prerequisites (one-time, human, not agent)

1. Clone the repo and check out `develop`:
   ```bash
   git clone <repo> rqvae-level-aware && cd rqvae-level-aware
   git checkout develop && git pull
   ```

2. Set up a local Python 3.11 venv and install launcher deps:
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install --upgrade pip
   pip install 'sagemaker>=2.230,<3' 'boto3>=1.34'
   # Stage 0 checkpoint loader also needs torch + gin for local smoke tests:
   pip install 'torch>=2.5.1' 'torchvision>=0.20.1' --index-url https://download.pytorch.org/whl/cpu
   pip install gin-config==0.5.0 einops 'polars==1.9.0' sentence-transformers accelerate pandas
   ```

3. Populate `.env` (repo-local, gitignored) with your AWS identifiers:
   ```
   RQVAE_S3_BASE=s3://<your-bucket>/rqvae-level-aware
   RQVAE_SAGEMAKER_ROLE=arn:aws:iam::<acct-id>:role/<role-name>
   RQVAE_AWS_PROFILE=<your-profile>
   RQVAE_AWS_REGION=us-east-1
   ```
   Every shell in this runbook starts with `set -a; source .env; set +a`.

4. Refresh credentials (e.g. `mwinit`, `aws sso login --profile $RQVAE_AWS_PROFILE`).

5. Fetch the upstream RQ-VAE checkpoints into `trained_models/` (gitignored):
   ```bash
   git remote add upstream https://github.com/EdoardoBotta/RQ-VAE-Recommender.git 2>/dev/null || true
   git fetch upstream main
   git checkout upstream/main -- trained_models/
   ls trained_models/rqvae_amazon_beauty trained_models/rqvae_amazon_sports trained_models/rqvae_ml32m
   ```

From here on, an agent can drive.

---

## Stage 0 — pipeline gate

### 0.1 Local smoke check — checkpoints + ItemData

```bash
set -a; source .env; set +a
PYTHONPATH=. python scripts/stage0_pipeline_check.py
cat docs/paper_plan_stage0_report.md
```

The script writes `docs/paper_plan_stage0_report.md` with a pass/fail verdict
per (dataset × checkpoint-load × data-pipeline). Exit code 0 iff every
requested dataset is green.

**Decision tree based on the report:**

- All three green → proceed to Stage 0.2.
- Beauty + Sports green, ML32M red on `data_loads` → record the error,
  treat as an ML32M pipeline-compat issue, then do **one** of:
  - Fix the issue in `data/ml32m.py` (max one working day), re-run 0.1.
  - Drop ML32M from the plan — edit `docs/paper_plan.md` and `AGENTS.md`'s
    dataset table to strike it through, proceed with Beauty + Sports only.
- Any dataset red on `checkpoint_present` → the upstream sync in step
  §0 didn't land that file. Re-run the `git checkout upstream/main --`
  command or fetch from HuggingFace Hub.

### 0.2 Cloud checkpoint validator — codebook fingerprints

Launch a validator job per dataset on SageMaker. The launcher's
`TARGETS` map already contains the upstream-pointing entries; if not, add
them following the pattern in `sagemaker/launch/launch_validate_rqvae.py`.

```bash
set -a; source .env; set +a
# Upload upstream checkpoints to S3 once (skip if already done):
for d in beauty sports ml32m; do
  case $d in
    beauty) local=trained_models/rqvae_amazon_beauty/checkpoint_high_entropy.pt ;;
    sports) local=trained_models/rqvae_amazon_sports/checkpoint_high_entropy.pt ;;
    ml32m)  local=trained_models/rqvae_ml32m/checkpoint_high_entropy.pt ;;
  esac
  key="checkpoints/rqvae_${d}_upstream/checkpoint_high_entropy.pt"
  aws s3 cp "$local" "$RQVAE_S3_BASE/$key" --profile "$RQVAE_AWS_PROFILE"
done
```

Then, for each dataset, add a `{dataset}_upstream` target to
`launch_validate_rqvae.py::TARGETS` pointing at
`checkpoints/rqvae_{dataset}_upstream/checkpoint_high_entropy.pt` and run:

```bash
python sagemaker/launch/launch_validate_rqvae.py \
    --targets beauty_upstream sports_upstream ml32m_upstream
```

Capture the three job names; monitor with §M1 (see bottom).

**Success:** each `verdict.json` either reports `healthy: true` (Beauty
expected) or a partial-collapse fingerprint with `unique_sids ≥ 10` and
`min_codebook_distance ≥ 0.01` (Sports, maybe ML32M). Sports fingerprint
is the already-known (33, 96, 134) / (1.78, 3.81, 4.82 bits) pattern and
is **accepted**.

Record the three fingerprints as a table in
`docs/paper_plan_stage0_report.md` — they become the paper's "setup"
section.

---

## Stage 1 — vanilla decoder + alpha-free beam strategies

For each dataset that cleared Stage 0:

### 1.1 Launch decoder training

```bash
set -a; source .env; set +a
# Each dataset uses a distinct upstream RQ-VAE, so launch per-dataset.
for d in beauty sports ml32m; do
    python sagemaker/launch/launch_decoder.py --datasets "$d" \
        --pretrained-rqvae "$RQVAE_S3_BASE/checkpoints/rqvae_${d}_upstream/checkpoint_high_entropy.pt"
done
```

Each launches a spot `ml.g5.2xlarge` training. Expect ~12–20 h wall-clock
per dataset. Monitor with §M1.

### 1.2 Eval sweep — alpha-free strategies

Once the three decoder checkpoints have landed in
`$RQVAE_S3_BASE/decoder/<dataset>/<job>/output/model.tar.gz`, run the
alpha-free eval sweep. `launch_decoding_eval.py` fans out to
`cardinality(--datasets) × cardinality(--strategies)` jobs and
auto-discovers the most recently modified decoder + RQ-VAE ckpts per
dataset.

```bash
set -a; source .env; set +a

# 4 alpha-free strategies × 3 datasets = 12 jobs.
python sagemaker/launch/launch_decoding_eval.py \
    --datasets beauty sports ml32m \
    --strategies vanilla dbs gumbel_topk hybrid \
    --decoder-variant baseline
```

Each job is ~1 h on `ml.g5.xlarge` spot. Monitor with §M2 (filter on
`eval-baseline-` or just `eval-`).

**`sasrec_rerank` note**: this strategy needs a SASRec aux head, so run it
against the **Stage 2 MTL decoder** once Stage 2.1 completes. The
launcher will auto-upgrade `--decoder-variant` to `mtl` when you ask for
`sasrec_rerank` (or any level-aware variant):

```bash
# After Stage 2.1 has MTL decoders ready:
python sagemaker/launch/launch_decoding_eval.py \
    --datasets beauty sports ml32m \
    --strategies sasrec_rerank
```

### 1.3 Aggregate Stage 1 results

```bash
PYTHONPATH=. python scripts/collect_results.py \
    --prefix rqvae-level-aware/eval-results/baseline \
    --aggregate-output results/stage1/all_runs.parquet \
    --per-user-output results/stage1/per_user_runs.parquet
```

(Bucket defaults to the one encoded in `$RQVAE_S3_BASE`.)

Confirm the resulting parquet has 4 strategies × 3 datasets = 12 rows.
Add `sasrec_rerank` rows once its MTL-decoder eval lands (step 1.2
second invocation).

---

## Stage 2 — MTL decoder + per-level alpha sweep

### 2.1 Launch MTL training

```bash
set -a; source .env; set +a
for d in beauty sports ml32m; do
    python sagemaker/launch/launch_mtl.py --datasets "$d" \
        --pretrained-rqvae "$RQVAE_S3_BASE/checkpoints/rqvae_${d}_upstream/checkpoint_high_entropy.pt"
done
```

Three spot `ml.g5.2xlarge` trainings. Expect ~15–24 h wall-clock per dataset.
Monitor with §M1.

Note: Stage 2 can start in parallel with Stage 1's eval sweep — the two
don't share checkpoints. Only Stage 1's `sasrec_rerank` eval depends on
Stage 2 being done.

### 2.2 Alpha pilot — 3³ grid per dataset

Each alpha-search job builds the MTL decoder + RQ-VAE once and iterates
through the entire alpha grid internally — one SageMaker job per dataset,
not per alpha point. MTL decoder + RQ-VAE checkpoints are auto-discovered.

```bash
set -a; source .env; set +a
python sagemaker/launch/launch_alpha_search.py \
    --datasets beauty sports ml32m \
    --alpha-grid "0.0,0.5,1.0" \
    --job-suffix pilot
```

3 jobs, each iterating 27 alpha points (3³). Expect ~2–4 h per job on
`ml.g5.xlarge` spot. Monitor with §M2 (filter on `alpha-` name prefix).

Once each job completes, pull its output CSV and pick the pilot winner:

```bash
for d in beauty sports ml32m; do
    aws s3 cp --recursive \
        "$RQVAE_S3_BASE/alpha-search/${d}-pilot/" \
        "results/stage2/alpha_${d}_pilot/" \
        --profile "$RQVAE_AWS_PROFILE"
    python -c "
import pandas as pd, glob, os
csv = sorted(glob.glob('results/stage2/alpha_${d}_pilot/**/*.csv', recursive=True))[-1]
df = pd.read_csv(csv).sort_values('recall_at_10', ascending=False)
print('${d} pilot winner:', df.iloc[0].to_dict())
"
done
```

Record each dataset's pilot winner for step 2.3.

### 2.3 Alpha refine — 5³ centred on pilot winner

Per-dataset refined grid (step 0.1, clipped to [0,1]), example shown for
a hypothetical Beauty pilot winner of (0.5, 0.5, 0.0):

```bash
python sagemaker/launch/launch_alpha_search.py \
    --datasets beauty \
    --alpha0-grid "0.3,0.4,0.5,0.6,0.7" \
    --alpha1-grid "0.3,0.4,0.5,0.6,0.7" \
    --alpha2-grid "0.0,0.1,0.2,0.3,0.4" \
    --job-suffix refined
```

Run once per dataset with grids centred on that dataset's pilot winner.
3 jobs, each iterating 125 alpha points (5³). Expect ~6–10 h per job on
`ml.g5.xlarge` spot.

### 2.4 Learned alpha — deferred (requires a launcher)

`scripts/train_alpha_params.py` is the reference implementation but does
not currently have a SageMaker launcher wired to it. To include the
learned-α row in the paper, one of:

- Run `scripts/train_alpha_params.py` locally against the downloaded MTL
  checkpoint (~4 h on a single GPU machine if you have one).
- Add a `launch_alpha_params.py` launcher paralleling `launch_alpha_search.py`
  (small addition; out of scope for this runbook but trivial for the agent
  to author if the paper plan demands it).

Skip this step if the refined-grid winner satisfies the paper story.

### 2.5 Aggregate Stage 2 results

```bash
PYTHONPATH=. python scripts/collect_results.py \
    --prefix rqvae-level-aware/alpha-search \
    --aggregate-output results/stage2/all_runs.parquet \
    --per-user-output results/stage2/per_user_runs.parquet
```

Then produce paper artefacts:

```bash
PYTHONPATH=. python scripts/make_tables.py --stage 1 --stage 2 \
    --output paper/tables/
PYTHONPATH=. python scripts/make_figures.py --stage 2 \
    --output paper/figures/
```

---

## Monitors

### §M1 — long training jobs (decoders, MTL)

```bash
set -a; source .env; set +a

jobs=(decoder-beauty decoder-sports decoder-ml32m \
      decoder-mtl-beauty decoder-mtl-sports decoder-mtl-ml32m)

while true; do
    done_count=0
    for job in "${jobs[@]}"; do
        status=$(aws sagemaker describe-training-job \
            --training-job-name "$job" \
            --profile "$RQVAE_AWS_PROFILE" \
            --region "$RQVAE_AWS_REGION" \
            --query TrainingJobStatus --output text 2>/dev/null || echo "NotFound")
        billing=$(aws sagemaker describe-training-job \
            --training-job-name "$job" \
            --profile "$RQVAE_AWS_PROFILE" \
            --region "$RQVAE_AWS_REGION" \
            --query BillableTimeInSeconds --output text 2>/dev/null || echo "-")
        printf "  %-28s  %-12s  billable=%ss\n" "$job" "$status" "$billing"
        case "$status" in
            Completed|Failed|Stopped|NotFound) done_count=$((done_count + 1)) ;;
        esac
    done
    echo "---"
    [ "$done_count" -eq "${#jobs[@]}" ] && break
    sleep 300
done
```

When a job reports `Failed`, grab the last 100 log lines for triage:

```bash
aws logs tail /aws/sagemaker/TrainingJobs \
    --log-stream-name-prefix decoder-mtl-ml32m/ \
    --profile "$RQVAE_AWS_PROFILE" --region "$RQVAE_AWS_REGION" \
    --since 6h | tail -100
```

### §M2 — many short eval jobs (alpha sweeps, strategy sweeps)

```bash
set -a; source .env; set +a

aws sagemaker list-training-jobs \
    --profile "$RQVAE_AWS_PROFILE" \
    --region "$RQVAE_AWS_REGION" \
    --max-results 200 \
    --status-equals InProgress \
    --name-contains alpha-search \
    --query 'TrainingJobSummaries[].[TrainingJobName,TrainingJobStatus]' \
    --output table

# Poll until all alpha-search jobs are terminal:
while true; do
    in_progress=$(aws sagemaker list-training-jobs \
        --profile "$RQVAE_AWS_PROFILE" --region "$RQVAE_AWS_REGION" \
        --max-results 200 --status-equals InProgress \
        --name-contains alpha-search \
        --query 'length(TrainingJobSummaries)' --output text)
    completed=$(aws sagemaker list-training-jobs \
        --profile "$RQVAE_AWS_PROFILE" --region "$RQVAE_AWS_REGION" \
        --max-results 500 --status-equals Completed \
        --name-contains alpha-search \
        --query 'length(TrainingJobSummaries)' --output text)
    failed=$(aws sagemaker list-training-jobs \
        --profile "$RQVAE_AWS_PROFILE" --region "$RQVAE_AWS_REGION" \
        --max-results 500 --status-equals Failed \
        --name-contains alpha-search \
        --query 'length(TrainingJobSummaries)' --output text)
    echo "$(date +%T)  in_progress=$in_progress  completed=$completed  failed=$failed"
    [ "$in_progress" = "0" ] && break
    sleep 120
done
```

Replace `--name-contains alpha-search` with `--name-contains eval-` for
the Stage 1 strategy sweep.

### §M3 — overall progress dashboard

Single-command snapshot of where the pipeline stands. Safe to re-run at
any time; does not launch anything new.

```bash
set -a; source .env; set +a
printf "\n== Stage 0: validator verdicts ==\n"
aws s3 ls "$RQVAE_S3_BASE/validate-rqvae/" --recursive \
    --profile "$RQVAE_AWS_PROFILE" | grep verdict.json || echo "  (none yet)"

printf "\n== Stage 1: vanilla decoder checkpoints ==\n"
aws s3 ls "$RQVAE_S3_BASE/decoder/" --recursive \
    --profile "$RQVAE_AWS_PROFILE" | grep model.tar.gz || echo "  (none yet)"

printf "\n== Stage 2: MTL decoder checkpoints ==\n"
aws s3 ls "$RQVAE_S3_BASE/decoder-mtl/" --recursive \
    --profile "$RQVAE_AWS_PROFILE" | grep model.tar.gz || echo "  (none yet)"

printf "\n== Stage 2: alpha-search outputs ==\n"
aws s3 ls "$RQVAE_S3_BASE/alpha-search/" --recursive \
    --profile "$RQVAE_AWS_PROFILE" | wc -l

printf "\n== Local aggregates ==\n"
ls -la results/stage*/all_runs.parquet 2>/dev/null || echo "  (not generated yet)"
```

---

## Final agent checklist (paste at end of a session)

Before closing out a run, the driving agent should post a single-paragraph
status update covering:

1. Which stages are **complete** (Stage 0 report exists? Stage 1 eval
   sweep parquet populated? Stage 2 refined-grid winner recorded?).
2. Which jobs are **in flight**, with job names.
3. Which jobs **failed** in this session, with the last error line.
4. Next action (e.g. "waiting on Stage 2 MTL training, ETA ~15 h").
5. Any deviation from this runbook, with the reason.

Then append that paragraph to `docs/progress_log.md` (create it if it
doesn't exist — each entry dated) so the next agent session picks up
with full context instead of re-discovering state.
