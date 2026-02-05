# IE7374 Lab Submissions
This repository contains my lab assignments.

## GitHub Action
Workflows defined in the `.github/workflows` folder.
### lab1
The code is located in the `GitHub_Action/lab1/` folder.
GitHub Actions is configured to run `lab1_script.py` automatically on every push.
### lab2
This workflow implements a daily scheduled check that triggers model validation only when updates are detected in the `GitHub_Action/lab2` directory.\
It utilizes a Python-based quality gate to enforce a 95% accuracy threshold, \
automatically failing the build and generating a visual performance report in the GitHub Action Summary if the model underperforms.

## Airflow ML Pipeline Lab - Iris Classification

A complete Apache Airflow pipeline for machine learning workflow orchestration, implementing Random Forest classification on the Iris dataset.

Project Overview

This project demonstrates how to build and orchestrate a machine learning pipeline using Apache Airflow running in Docker containers. The pipeline performs data loading, preprocessing, model training, and evaluation in a structured, reproducible workflow.

Learning Objectives

- Set up Apache Airflow using Docker Compose
- Create DAGs (Directed Acyclic Graphs) for ML workflows
- Implement task dependencies and data passing via XCom
- Monitor and debug pipeline execution
- Save and version trained ML models

Pipeline Architecture
```
load_data → preprocess_data → train_model → evaluate_model
```

**Task Breakdown:**
1. **load_data**: Loads Iris dataset from scikit-learn
2. **preprocess_data**: Performs train/test split (80/20) and feature scaling
3. **train_model**: Trains Random Forest classifier and saves model
4. **evaluate_model**: Evaluates model on test set and logs metrics





Project Structure
```
airflow-lab/
├── dags/
│   └── iris_classification_dag.py    # Airflow DAG definition
├── src/
│   ├── __init__.py                   # Python package marker
│   └── ml_functions.py               # ML pipeline functions
├── models/
│   └── iris_rf_model.pkl             # Trained Random Forest model
├── logs/                             # Airflow task logs
├── plugins/                          # Airflow plugins (empty)
├── working_data/                     # Temporary data storage
├── docker-compose.yaml               # Airflow services configuration
├── .env                              # Environment variables
└── README.md                         # This file
```