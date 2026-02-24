# Airflow Lab — Iris Classification Pipeline

## What This Lab Does

Orchestrates a 4-task ML pipeline using Apache Airflow running in Docker. Each task is a `PythonOperator` that passes data to the next task via **XCom** (Airflow's built-in cross-task communication). The pipeline trains a Random Forest classifier on the Iris dataset and logs a full evaluation report.

```
load_data → preprocess_data → train_model → evaluate_model
```

## Key Concepts

- **DAG**: defines the task graph and execution order; manual trigger only (`schedule_interval=None`)
- **PythonOperator**: wraps a Python function as an Airflow task
- **XCom with pickle**: tasks return serialized data (`pickle.dumps`) which Airflow stores in XCom; downstream tasks retrieve it via `ti.xcom_pull(task_ids='...')`. Pickling is enabled via `AIRFLOW__CORE__ENABLE_XCOM_PICKLING=true` in `.env`
- **Task isolation**: each function only knows its own inputs and outputs — no shared global state

## Task Breakdown

| Task | Function | What it does |
|---|---|---|
| `load_data` | `load_data()` | Loads Iris from sklearn, serializes to XCom |
| `preprocess_data` | `data_preprocessing()` | 80/20 stratified split + StandardScaler |
| `train_model` | `build_save_model()` | Trains RandomForest, saves `.pkl` to `/opt/airflow/models/` |
| `evaluate_model` | `evaluate_model()` | Loads model, prints accuracy + confusion matrix + classification report |

## How to Run

```bash
# 1. Start Airflow
cd airflow_lab
docker-compose up airflow-init
docker-compose up

# 2. Open the UI
# http://localhost:8080  (admin / admin123)

# 3. Trigger the DAG manually
# Find 'iris_classification_pipeline' and click Run
```

## Project Structure

```
airflow_lab/
├── dags/
│   └── iris_classification_dag.py   # DAG definition and task wiring
├── src/
│   ├── __init__.py
│   └── ml_functions.py              # load, preprocess, train, evaluate
├── models/
│   └── iris_rf_model.pkl            # Saved model (generated at runtime)
├── docker-compose.yaml
├── .env                             # AIRFLOW_UID, XCom pickling flag
└── README.md
```