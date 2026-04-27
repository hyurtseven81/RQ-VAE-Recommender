# Runbook index — CIKM 2026 experimental pipeline

The experimental pipeline is split into two side-by-side runbooks.
Both assume the repo is already cloned and pulled to the current
working directory. Hand each off to a separate automation agent run.

| File | Scope | When to run |
|---|---|---|
| `docs/runbook_operator.md`     | venv + AWS auth setup, Stage 0 inventory + ML32M pipeline gate, post-SageMaker result aggregation, paper tables/figures | First (Stage 0 gate) and last (aggregation + artefacts) |
| `docs/runbook_sagemaker.md` | upstream RQ-VAE validation, vanilla decoder + alpha-free eval sweep, MTL decoder + alpha pilot/refined/learned + their eval sub-runs, sasrec_rerank | Between the two local runs |

Bisect material for the parked codebook-collapse investigation lives
in `docs/bisect_runbook.md` and is **not** part of the paper-critical
path.

## Suggested order

1. `docs/runbook_operator.md` §0 + §1 — pre-flight + Stage 0 gate.
   - Output: `docs/paper_plan_stage0_report.md` decides whether ML32M
     is in scope.
2. `docs/runbook_sagemaker.md` §0 → §3 — every SageMaker launch + wait.
   - Stage 1 decoder training and Stage 2 MTL training run in
     parallel; everything else gates on its predecessor as documented
     inside that file.
3. `docs/runbook_operator.md` §3 + §4 — pull result tarballs from S3,
   build `results/all_runs.parquet`, render `paper/tables/` and
   `paper/figures/`.

Both runbooks end by appending a dated paragraph to
`docs/progress_log.md` so the next session has full context.
