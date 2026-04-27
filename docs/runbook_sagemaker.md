# SageMaker runbook — cloud-side launches for the CIKM 2026 pipeline

You are an automation agent. Follow this runbook to drive every
SageMaker training and evaluation job for the paper. The repo is
already cloned and pulled to the current working directory; do not
re-clone.

This runbook does not run anything compute-heavy locally. Local-only
steps (Stage 0 gate, aggregation, paper artefacts) live in
`docs/runbook_operator.md`. **Run the operator runbook's §0 (pre-flight) and
§1 (Stage 0 gate) before this one** — they decide whether ML32M is in
scope and whether it is safe to spend SageMaker quota.

The experimental scope is `docs/paper_plan.md`; this runbook is the
"how", that file is the "why".

## Conventions for the agent

- Run every command from the repo root unless stated otherwise.
- Start each shell invocation with `set -a; source .env; set +a` so
  `RQVAE_S3_BASE`, `RQVAE_SAGEMAKER_ROLE`, `RQVAE_AWS_PROFILE`,
  `RQVAE_AWS_REGION` are loaded.
- `launch_*.py` scripts call `estimator.fit(..., wait=False, logs=False)`
  — they return immediately. The polling loops in §M1 / §M2 are how
  you wait.
- Never edit the gin configs or launchers without an explicit
  instruction from the operator. If a command fails, stop and report
  the failing command + the last 30 lines of stdout/stderr.
- Cap polling loops at the documented timeouts. If a job exceeds them,
  stop polling and ask the operator how to proceed.
- Capture every launched job name into `/tmp/sagemaker_jobs.log` so
  the final report can list them.
- Append a dated paragraph to `docs/progress_log.md` (create if
  needed) at the end of the session summarising what ran, what is
  still in flight, and what failed.

## §0 Pre-flight

### 0.1 Env

```bash
test -f .env || { echo "MISSING: .env"; exit 1; }
awk -F= '/^[[:space:]]*RQVAE_(S3_BASE|SAGEMAKER_ROLE|AWS_PROFILE|AWS_REGION)[[:space:]]*=/ \
    {print $1 ": SET"}' .env
```

If any of the four are not `: SET`, ask the operator to populate `.env`
using this template (do not echo their values):

```
RQVAE_S3_BASE=s3://<bucket>/rqvae-level-aware
RQVAE_SAGEMAKER_ROLE=arn:aws:iam::<account-id>:role/<role-name>
RQVAE_AWS_PROFILE=<boto3-profile-name>
RQVAE_AWS_REGION=us-east-1
```

### 0.2 Auth + venv

```bash
test -d .venv || python -m venv .venv
source .venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet 'sagemaker>=2.230,<3' 'boto3>=1.34'

# Refresh AWS credentials (try whichever applies):
mwinit                                          # Midway / Amazon
aws sso login --profile "$RQVAE_AWS_PROFILE"    # AWS SSO

# Sanity:
aws sts get-caller-identity --profile "$RQVAE_AWS_PROFILE"
```

### 0.3 Local trained_models/ presence

`source_dir="."` uploads the entire repo to SageMaker, including the
gitignored `trained_models/`. Without these checkpoints locally, the
training jobs that don't pass `--pretrained-rqvae` will fail at gin
load time.

```bash
ls trained_models/rqvae_amazon_beauty trained_models/rqvae_amazon_sports trained_models/rqvae_ml32m
```

If any directory is missing, run §0.4 of `docs/runbook_operator.md` to
sync from upstream and come back.

### 0.4 Stage 0 verdict

Confirm the local Stage 0 gate has been run and recorded:

```bash
ls -l docs/paper_plan_stage0_report.md
```

Read the report and capture the dataset list from its verdict table.
If ML32M is **BLOCKED**, drop it from every `--datasets` flag in the
rest of this runbook. The operator should have already noted this in
`docs/progress_log.md`. From here on this runbook uses
`DATASETS=(beauty sports ml32m)` — replace with `(beauty sports)` if
ML32M was dropped.

Status: `§0: PASS` once env, auth, ckpts, and Stage 0 verdict are all
clear.

## §1 Stage 0.2 — upstream RQ-VAE validation on SageMaker

Goal: produce a `verdict.json` per upstream checkpoint so the paper's
"setup" section can cite per-level codebook fingerprints.

### 1.1 Upload upstream checkpoints to S3 (idempotent)

The validator launcher's `*_upstream` targets read from
`$RQVAE_S3_BASE/checkpoints/rqvae_<dataset>_upstream/`. Upload each
local file there if missing.

```bash
set -a; source .env; set +a
for d in beauty sports ml32m; do
    case $d in
        beauty) local=trained_models/rqvae_amazon_beauty/checkpoint_high_entropy.pt ;;
        sports) local=trained_models/rqvae_amazon_sports/checkpoint_high_entropy.pt ;;
        ml32m)  local=trained_models/rqvae_ml32m/checkpoint_high_entropy.pt ;;
    esac
    key="checkpoints/rqvae_${d}_upstream/checkpoint_high_entropy.pt"
    bucket="${RQVAE_S3_BASE#s3://}"
    bucket_only="${bucket%%/*}"
    prefix="${bucket#*/}"
    full_key="${prefix}/${key}"
    if aws s3api head-object \
            --bucket "$bucket_only" --key "$full_key" \
            --profile "$RQVAE_AWS_PROFILE" --region "$RQVAE_AWS_REGION" \
            >/dev/null 2>&1; then
        echo "$d: already on S3"
    else
        aws s3 cp "$local" "$RQVAE_S3_BASE/$key" \
            --profile "$RQVAE_AWS_PROFILE" --region "$RQVAE_AWS_REGION"
    fi
done
```

### 1.2 Launch validator jobs

```bash
python sagemaker/launch/launch_validate_rqvae.py \
    --targets beauty_upstream sports_upstream ml32m_upstream \
    2>&1 | tee -a /tmp/sagemaker_jobs.log
```

Capture the three job names from the `Launched: <name>` lines.

### 1.3 Wait for completion

Use §M1 (polling loop at the bottom of this file) with the three
validator jobs. Validator runs are short (~20–60 min on
`ml.g5.4xlarge` on-demand), all 3 should be `Completed` within ~1 h.

### 1.4 Pull verdicts

```bash
mkdir -p results/stage0/validators
for stub in validate-rqvae-beauty-upstream \
            validate-rqvae-sports-upstream \
            validate-rqvae-ml32m-upstream; do
    job_full=$(grep -oE "${stub}-[0-9]{8}-[0-9]{6}" /tmp/sagemaker_jobs.log | tail -1)
    [ -z "$job_full" ] && { echo "no job for $stub"; continue; }
    aws s3 cp \
        "$RQVAE_S3_BASE/validate-rqvae/${stub}/${job_full}/output/output.tar.gz" \
        "results/stage0/validators/${stub}.tar.gz" \
        --profile "$RQVAE_AWS_PROFILE" --region "$RQVAE_AWS_REGION" || \
    aws s3 cp \
        "$RQVAE_S3_BASE/validate-rqvae/${stub}/${job_full}/output/model.tar.gz" \
        "results/stage0/validators/${stub}.tar.gz" \
        --profile "$RQVAE_AWS_PROFILE" --region "$RQVAE_AWS_REGION"
    tar -xzf "results/stage0/validators/${stub}.tar.gz" \
        -C "results/stage0/validators/" verdict.json 2>/dev/null && \
        mv results/stage0/validators/verdict.json \
           "results/stage0/validators/${stub}.verdict.json"
    cat "results/stage0/validators/${stub}.verdict.json" 2>/dev/null
done
```

Record the per-level fingerprint for each dataset in
`docs/paper_plan_stage0_report.md` under a "Cloud fingerprints"
section.

Status: `§1: PASS` once every requested target's `verdict.json` exists
and shows `unique_sids ≥ 10` and `min_codebook_distance ≥ 0.01` per
level. A `healthy: false` flag with those minimums met is acceptable
(Sports is known partial-collapse; see AGENTS.md datasets table).

## §2 Stage 1 — vanilla decoder + alpha-free strategy sweep

### 2.1 Launch decoder training (one job per dataset)

Each training is ~12–20 h on `ml.g5.2xlarge` spot. They can run in
parallel.

```bash
set -a; source .env; set +a
DATASETS=(beauty sports ml32m)   # drop ml32m if Stage 0 blocked it
for d in "${DATASETS[@]}"; do
    python sagemaker/launch/launch_decoder.py --datasets "$d" \
        --pretrained-rqvae "$RQVAE_S3_BASE/checkpoints/rqvae_${d}_upstream/checkpoint_high_entropy.pt" \
        2>&1 | tee -a /tmp/sagemaker_jobs.log
done
```

Capture the launched job names. Stage 2.1 (MTL training) can run in
parallel with this — kick it off now from §3.1 below.

### 2.2 Wait for the three decoder jobs

Use §M1. Don't proceed to §2.3 until all three (or two if ML32M was
dropped) are `Completed`.

### 2.3 Launch the alpha-free eval sweep

4 alpha-free strategies × N datasets. The launcher auto-discovers the
most recently modified decoder ckpt under
`$RQVAE_S3_BASE/decoder/<dataset>/`. Each job is ~1 h on
`ml.g5.xlarge` spot.

```bash
python sagemaker/launch/launch_decoding_eval.py \
    --datasets "${DATASETS[@]}" \
    --strategies vanilla dbs gumbel_topk hybrid \
    --decoder-variant baseline \
    2>&1 | tee -a /tmp/sagemaker_jobs.log
```

### 2.4 Wait for the eval sweep

Use §M2 with `--name-contains eval-baseline-`. 4 × N jobs total.

`sasrec_rerank` is intentionally not in this sweep — it needs the
Stage 2 MTL decoder; it's launched separately in §3.5.

Status: `§2: PASS` once all (4 × N) Stage-1 eval jobs are `Completed`.

## §3 Stage 2 — MTL decoder + level-aware alpha sweeps

### 3.1 Launch MTL training (one job per dataset)

Can run in parallel with §2.1. Each is ~15–24 h on `ml.g5.2xlarge`
spot.

```bash
set -a; source .env; set +a
for d in "${DATASETS[@]}"; do
    python sagemaker/launch/launch_mtl.py --datasets "$d" \
        --pretrained-rqvae "$RQVAE_S3_BASE/checkpoints/rqvae_${d}_upstream/checkpoint_high_entropy.pt" \
        2>&1 | tee -a /tmp/sagemaker_jobs.log
done
```

### 3.2 Wait for the MTL jobs

Use §M1.

### 3.3 Alpha pilot grid (3³ per dataset)

One job per dataset. Each iterates 27 alpha points internally
(model is loaded once); ~2–4 h on `ml.g5.xlarge` spot.

```bash
python sagemaker/launch/launch_alpha_search.py \
    --datasets "${DATASETS[@]}" \
    --alpha-grid "0.0,0.5,1.0" \
    --job-suffix pilot \
    2>&1 | tee -a /tmp/sagemaker_jobs.log
```

Wait via §M2 (`--name-contains alpha-`). Then download the pilot
CSVs and identify the winner per dataset:

```bash
mkdir -p /tmp/alpha-pilot
for d in "${DATASETS[@]}"; do
    aws s3 sync "$RQVAE_S3_BASE/alpha-search/${d}-pilot/" \
        "/tmp/alpha-pilot/${d}/" \
        --profile "$RQVAE_AWS_PROFILE" --region "$RQVAE_AWS_REGION" \
        --quiet
    find "/tmp/alpha-pilot/${d}/" -name 'output.tar.gz' -exec sh -c '
        tar -xzf "$1" -C "$(dirname "$1")"
    ' _ {} \;
    csv=$(find "/tmp/alpha-pilot/${d}/" -name '*.csv' | head -1)
    [ -z "$csv" ] && { echo "$d: no CSV"; continue; }
    python -c "
import pandas as pd
df = pd.read_csv('$csv')
key = 'recall@10' if 'recall@10' in df.columns else 'ndcg@10'
best = df.sort_values(key, ascending=False).iloc[0]
triple = (best['alpha_0'], best['alpha_1'], best['alpha_2'])
print(f'{\"$d\"} pilot winner: alpha=({triple[0]:.2f},{triple[1]:.2f},{triple[2]:.2f}) {key}={best[key]:.4f}')
"
done
```

Record the pilot winner per dataset (you'll feed it into §3.4).

### 3.4 Alpha refined grid (5³ centred on each pilot winner)

Per dataset. Each refined grid is 125 points; ~6–10 h on
`ml.g5.xlarge` spot. Compute the per-level grid as `[winner-0.2,
winner-0.1, winner, winner+0.1, winner+0.2]` clipped to [0, 1].

```bash
# Example for beauty with pilot winner (0.5, 0.5, 0.0).
# Replace the three grid strings per dataset with your computed values.
python sagemaker/launch/launch_alpha_search.py \
    --datasets beauty \
    --alpha0-grid "0.3,0.4,0.5,0.6,0.7" \
    --alpha1-grid "0.3,0.4,0.5,0.6,0.7" \
    --alpha2-grid "0.0,0.1,0.2,0.3,0.4" \
    --job-suffix refined \
    2>&1 | tee -a /tmp/sagemaker_jobs.log

# Repeat for sports and ml32m with their respective pilot winners.
```

A small helper to clip a winner to a 5-point ±0.1 grid:

```bash
clip_grid () {
    # usage: clip_grid 0.5  ->  prints "0.3,0.4,0.5,0.6,0.7"
    python -c "
w = $1
g = [max(0.0, min(1.0, round(w + d, 2))) for d in (-0.2, -0.1, 0.0, 0.1, 0.2)]
print(','.join(f'{x:.1f}' for x in g))
"
}
```

Wait via §M2.

### 3.5 Launch the level-aware eval sub-runs (per dataset)

Three eval rows per dataset, one per alpha source. Each is ~1 h on
`ml.g5.xlarge` spot.

```bash
set -a; source .env; set +a

for d in "${DATASETS[@]}"; do
    # (a) level_aware_mix with refined-grid winner — pass alpha as comma-string.
    #     Replace <winner-triple> with the actual best from §3.4 per dataset.
    python sagemaker/launch/launch_decoding_eval.py \
        --datasets "$d" \
        --strategies level_aware_mix \
        --alpha "<winner-triple>" \
        2>&1 | tee -a /tmp/sagemaker_jobs.log

    # (b) level_aware_mix_grid — point at the refined CSV in S3.
    refined_csv_uri=$(aws s3 ls --recursive \
        "$RQVAE_S3_BASE/alpha-search/${d}-refined/" \
        --profile "$RQVAE_AWS_PROFILE" --region "$RQVAE_AWS_REGION" \
        | awk '/output.tar.gz/ {print $NF}' | tail -1)
    if [ -n "$refined_csv_uri" ]; then
        python sagemaker/launch/launch_decoding_eval.py \
            --datasets "$d" \
            --strategies level_aware_mix_grid \
            --alpha-csv "${RQVAE_S3_BASE%/*}/${refined_csv_uri%/output*}/output/output.tar.gz" \
            2>&1 | tee -a /tmp/sagemaker_jobs.log
    fi

    # (c) level_aware_mix_learned — point at the learned .pt in S3 (after §3.6).
done
```

Wait via §M2.

### 3.6 Learned alpha (one job per dataset)

```bash
python sagemaker/launch/launch_alpha_params.py \
    --datasets "${DATASETS[@]}" \
    --init-alpha "0.5,0.5,0.5" \
    --n-epochs 1 \
    --job-suffix learned \
    2>&1 | tee -a /tmp/sagemaker_jobs.log
```

On-demand `ml.g5.2xlarge` (spot disabled by default). ~1–4 h per
dataset. Wait via §M1 (`--name-contains alpha-learned-`).

Then run the learned-α eval sub-run:

```bash
for d in "${DATASETS[@]}"; do
    learned_uri=$(aws s3 ls --recursive \
        "$RQVAE_S3_BASE/alpha-learned/${d}-learned/" \
        --profile "$RQVAE_AWS_PROFILE" --region "$RQVAE_AWS_REGION" \
        | awk '/model.tar.gz/ {print $NF}' | tail -1)
    [ -z "$learned_uri" ] && { echo "$d: no learned ckpt"; continue; }
    python sagemaker/launch/launch_decoding_eval.py \
        --datasets "$d" \
        --strategies level_aware_mix_learned \
        --alpha-ckpt "s3://${RQVAE_S3_BASE#s3://}/${learned_uri}" \
        2>&1 | tee -a /tmp/sagemaker_jobs.log
done
```

### 3.7 sasrec_rerank against the MTL decoder

```bash
python sagemaker/launch/launch_decoding_eval.py \
    --datasets "${DATASETS[@]}" \
    --strategies sasrec_rerank \
    2>&1 | tee -a /tmp/sagemaker_jobs.log
```

The launcher auto-upgrades `--decoder-variant` to `mtl` because
`sasrec_rerank` requires the SASRec aux head. Wait via §M2.

Status: `§3: PASS` once §3.5, §3.6, and §3.7 jobs are all
`Completed`. Hand off to the operator runbook §3 for aggregation.

## §M Monitors

These are stand-alone polling loops. Each safely re-runs at any time.

### §M1 — long training jobs

```bash
set -a; source .env; set +a

# Replace this list with your launched job names from /tmp/sagemaker_jobs.log
jobs=(
    decoder-beauty decoder-sports decoder-ml32m
    decoder-mtl-beauty decoder-mtl-sports decoder-mtl-ml32m
)

while true; do
    done_count=0
    for job in "${jobs[@]}"; do
        status=$(aws sagemaker describe-training-job \
            --training-job-name "$job" \
            --profile "$RQVAE_AWS_PROFILE" --region "$RQVAE_AWS_REGION" \
            --query TrainingJobStatus --output text 2>/dev/null || echo NotFound)
        billable=$(aws sagemaker describe-training-job \
            --training-job-name "$job" \
            --profile "$RQVAE_AWS_PROFILE" --region "$RQVAE_AWS_REGION" \
            --query BillableTimeInSeconds --output text 2>/dev/null || echo 0)
        printf "  %-32s  %-12s  billable=%ss\n" "$job" "$status" "$billable"
        case "$status" in
            Completed|Failed|Stopped|NotFound) done_count=$((done_count + 1)) ;;
        esac
    done
    echo "---"
    [ "$done_count" -eq "${#jobs[@]}" ] && break
    sleep 300
done
```

If any job reports `Failed`, fetch the last 100 log lines for triage:

```bash
aws logs tail /aws/sagemaker/TrainingJobs \
    --log-stream-name-prefix <job-name>/ \
    --profile "$RQVAE_AWS_PROFILE" --region "$RQVAE_AWS_REGION" \
    --since 6h | tail -100
```

### §M2 — many short jobs (alpha sweeps, eval strategy sweep)

Polls by name prefix. Adjust `--name-contains` per use:

- alpha pilot/refined: `--name-contains alpha-`
- alpha learned: `--name-contains alpha-learned-`
- baseline eval: `--name-contains eval-baseline-`
- mtl eval: `--name-contains eval-mtl-`

```bash
set -a; source .env; set +a

NAME_FILTER="alpha-"   # change me

while true; do
    in_progress=$(aws sagemaker list-training-jobs \
        --profile "$RQVAE_AWS_PROFILE" --region "$RQVAE_AWS_REGION" \
        --max-results 200 --status-equals InProgress \
        --name-contains "$NAME_FILTER" \
        --query 'length(TrainingJobSummaries)' --output text)
    completed=$(aws sagemaker list-training-jobs \
        --profile "$RQVAE_AWS_PROFILE" --region "$RQVAE_AWS_REGION" \
        --max-results 500 --status-equals Completed \
        --name-contains "$NAME_FILTER" \
        --query 'length(TrainingJobSummaries)' --output text)
    failed=$(aws sagemaker list-training-jobs \
        --profile "$RQVAE_AWS_PROFILE" --region "$RQVAE_AWS_REGION" \
        --max-results 500 --status-equals Failed \
        --name-contains "$NAME_FILTER" \
        --query 'length(TrainingJobSummaries)' --output text)
    echo "$(date +%T)  in_progress=$in_progress  completed=$completed  failed=$failed"
    [ "$in_progress" = "0" ] && break
    sleep 120
done
```

### §M3 — overall progress dashboard

A snapshot you can run any time, doesn't launch anything new.

```bash
set -a; source .env; set +a

printf "\n== Stage 0: validator verdicts ==\n"
aws s3 ls "$RQVAE_S3_BASE/validate-rqvae/" --recursive \
    --profile "$RQVAE_AWS_PROFILE" --region "$RQVAE_AWS_REGION" \
    | grep -E 'verdict.json|output.tar.gz' | tail -20

printf "\n== Stage 1: vanilla decoder checkpoints ==\n"
aws s3 ls "$RQVAE_S3_BASE/decoder/" --recursive \
    --profile "$RQVAE_AWS_PROFILE" --region "$RQVAE_AWS_REGION" \
    | grep model.tar.gz

printf "\n== Stage 1 + 2: eval results ==\n"
aws s3 ls "$RQVAE_S3_BASE/eval-results/" --recursive \
    --profile "$RQVAE_AWS_PROFILE" --region "$RQVAE_AWS_REGION" \
    | grep output.tar.gz | wc -l
echo "(eval tarballs)"

printf "\n== Stage 2: MTL decoder checkpoints ==\n"
aws s3 ls "$RQVAE_S3_BASE/decoder-mtl/" --recursive \
    --profile "$RQVAE_AWS_PROFILE" --region "$RQVAE_AWS_REGION" \
    | grep model.tar.gz

printf "\n== Stage 2: alpha-search outputs ==\n"
aws s3 ls "$RQVAE_S3_BASE/alpha-search/" --recursive \
    --profile "$RQVAE_AWS_PROFILE" --region "$RQVAE_AWS_REGION" \
    | grep output.tar.gz | wc -l
echo "(alpha-search tarballs)"

printf "\n== Stage 2: learned alpha checkpoints ==\n"
aws s3 ls "$RQVAE_S3_BASE/alpha-learned/" --recursive \
    --profile "$RQVAE_AWS_PROFILE" --region "$RQVAE_AWS_REGION" \
    | grep model.tar.gz
```

## §F Final report

Append a dated paragraph to `docs/progress_log.md`. **Do not** include
S3 bucket names, account ids, profile names, role ARNs, or anything
that comes from `.env`. The file is gitignored but the operator may
copy excerpts elsewhere. The single-quoted heredoc below prevents the
shell from expanding any env vars.

```bash
cat <<'EOF' >> docs/progress_log.md

## __SESSION_TIMESTAMP__ — sagemaker runbook session

Stages launched:
  §1 validators:  <N jobs Completed | M Failed | … >
  §2 stage-1:     <N decoder + M eval jobs Completed | … >
  §3 stage-2:     <N MTL + M alpha-search + K eval jobs Completed | … >
Job log: /tmp/sagemaker_jobs.log
Open issues: <one-liner | none>
Next: <run docs/runbook_operator.md §3 (aggregation) | wait on …>
EOF

sed -i.bak "s|__SESSION_TIMESTAMP__|$(date -u +%FT%TZ)|" docs/progress_log.md \
    && rm -f docs/progress_log.md.bak
```

Reference S3 locations only as their `$RQVAE_S3_BASE/...` literal form
(the heredoc is single-quoted, so the dollar signs survive into the
file). Then post a two-line summary to the operator: which sections
passed, which (if any) jobs failed, and the next concrete step
(typically "run `docs/runbook_operator.md` §3").

> Reminder: `docs/progress_log.md` is gitignored on purpose (see
> `.gitignore`). Never `git add` it; if you do, scrub it first
> through the same lens used for source — no bucket / profile /
> account-id leakage.
