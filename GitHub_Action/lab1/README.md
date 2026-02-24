# Lab 1 — GitHub Actions: Automated Data Fetching

## What This Lab Does

This lab demonstrates how to use **GitHub Actions** to automatically run a data fetching script every time changes are pushed to the `GitHub_Action/lab1/` directory.

The script downloads the Iris dataset from a public URL, prints a data preview, summary statistics, and class distribution — then exits with a non-zero code if the download fails, causing GitHub Actions to correctly report the workflow as failed.

## Key Concepts

- **Path-filtered triggers**: the workflow only activates when files inside `GitHub_Action/lab1/` change, not on every push to the repo
- **Scoped dependencies**: the lab installs only what it needs (`pandas`) via its own `requirements.txt`, keeping CI fast
- **Proper failure propagation**: `sys.exit(1)` on error ensures GitHub reports failures instead of silently passing

## Workflow Trigger

```yaml
on:
  push:
    paths:
      - 'GitHub_Action/lab1/**'
```

Pushes to any other directory will **not** trigger this workflow.

## Pipeline Steps

```
Checkout → Setup Python 3.10 → Install Dependencies → Fetch and Validate Data
```

1. **Checkout**: pulls the repo into the runner
2. **Setup Python**: pins Python 3.10 for reproducibility
3. **Install Dependencies**: installs from `GitHub_Action/lab1/requirements.txt`
4. **Fetch and Validate Data**: runs `lab1_script.py`, which:
   - Downloads Iris CSV from `raw.githubusercontent.com`
   - Prints first 5 rows, descriptive statistics, and class distribution
   - Exits with code 1 on any failure so the workflow fails visibly

## Files

```
lab1/
├── lab1_script.py      # Data fetching and validation script
├── requirements.txt    # Scoped dependencies (pandas only)
└── README.md
```

## How to Test Locally

```bash
pip install -r GitHub_Action/lab1/requirements.txt
python GitHub_Action/lab1/lab1_script.py
```
