# Contributing

This is a research fork of [EdoardoBotta/RQ-VAE-Recommender](https://github.com/EdoardoBotta/RQ-VAE-Recommender), extending TIGER with level-aware hybrid decoding.

## Branch structure

| Branch | Purpose |
|--------|---------|
| `main` | Tracks upstream — never commit directly |
| `develop` | Integration branch — all feature branches target here |
| `feat/eval-harness` | Per-user metrics, NDCG@K, paired bootstrap |
| `feat/decoding-strategies` | DBS, Gumbel, hybrid, vanilla refactor |
| `feat/sasrec-head` | SASRec auxiliary head + MTL training |
| `feat/level-aware-mixing` | Per-level alpha mixing during beam expansion |
| `feat/sagemaker-infra` | SageMaker containers, estimators, Steam data |
| `feat/residual-analysis` | Residual entropy analysis and paper scripts |

## Syncing with upstream

```bash
git remote add upstream https://github.com/EdoardoBotta/RQ-VAE-Recommender.git
git fetch upstream
git checkout main
git merge upstream/main
git checkout develop
git rebase main
```

## Development setup

```bash
pip install -e ".[dev,experiments]"
pre-commit install
```

## Running tests

```bash
pytest tests/ -v
pytest tests/ --cov=modules/decoding --cov-report=term-missing
```

## SageMaker jobs

All SageMaker training and eval jobs run from the compute machine (not this dev machine), targeting `s3://YOUR_S3_BUCKET/rqvae-level-aware/` under the YOUR_AWS_PROFILE AWS profile.

From the compute machine:
```bash
export AWS_PROFILE=YOUR_AWS_PROFILE
python sagemaker/launch/launch_baselines.py --dataset beauty
```

## LIGER baseline

LIGER paper results are used directly from Yang et al. 2024 Table 1 (not reproduced). Stored in `results/baselines/liger_paper_numbers.json`. If cloning LIGER for reference: `git clone https://github.com/facebookresearch/liger ../liger`.
