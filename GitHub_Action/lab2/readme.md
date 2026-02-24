# Lab 2 — GitHub Actions: Scheduled Model Validation

## What This Lab Does

Runs an automated quality gate that trains and validates a RandomForest model on the Iris dataset, enforcing a **95% accuracy threshold**. Triggered on every push to `GitHub_Action/lab2/` and on a daily schedule (weekdays 9AM Beijing time), but the scheduled run **skips execution if no changes were made in the last 24 hours**.

Results are written to the GitHub Actions Job Summary as a Markdown report, showing pass or fail status clearly.

## Key Concepts

- **Conditional execution**: scheduled runs check git history first and skip if nothing changed
- **Separated run vs. report steps**: the validation step and summary step are decoupled so the report always writes, even on failure (`if: always()`)
- **Deterministic results**: fixed `random_state=42` on both the train/test split and the model ensures the same result every run — no flaky CI failures from random seeds
- **Clean script output**: `check_iris.py` prints only the accuracy value so the workflow can capture it cleanly into `$GITHUB_OUTPUT`

## Workflow Triggers

```yaml
on:
  schedule:
    - cron: '0 1 * * 1-5'   # weekdays only, skips if no recent changes
  push:
    paths:
      - 'GitHub_Action/lab2/**'
  workflow_dispatch:           # manual trigger via GitHub UI
```

## Pipeline Steps

```
Checkout → Check Changes → Setup Python → Install Deps → Run Validation → Write Summary
```

The last two steps use `if: always()` so the summary report is written even when validation fails.

## Files

```
lab2/
├── check_iris.py   # Trains model, prints accuracy, exits 1 if below threshold
└── README.md
```

## How to Test Locally

```bash
pip install scikit-learn
python GitHub_Action/lab2/check_iris.py
```