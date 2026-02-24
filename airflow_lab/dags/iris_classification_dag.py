"""
Airflow DAG for Iris Classification Pipeline
DAG: iris_classification_pipeline
Tasks: load_data -> preprocess_data -> train_model -> evaluate_model

XCom pickling is enabled via AIRFLOW__CORE__ENABLE_XCOM_PICKLING=true in .env
Data is passed between tasks using ti.xcom_pull inside each callable.
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
sys.path.insert(0, '/opt/airflow')
from src.ml_functions import load_data, data_preprocessing, build_save_model, evaluate_model

default_args = {
    'owner': 'data_scientist',
    'start_date': datetime(2026, 2, 5),
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
}

dag = DAG(
    'iris_classification_pipeline',
    default_args=default_args,
    description='ML Pipeline for Iris Classification using Random Forest',
    schedule_interval=None,
    catchup=False,
    tags=['machine_learning', 'classification', 'iris'],
)

load_data_task = PythonOperator(
    task_id='load_data',
    python_callable=load_data,
    dag=dag,
)

preprocess_task = PythonOperator(
    task_id='preprocess_data',
    python_callable=data_preprocessing,
    dag=dag,
)

train_model_task = PythonOperator(
    task_id='train_model',
    python_callable=build_save_model,
    op_kwargs={'model_filename': 'iris_rf_model.pkl'},
    dag=dag,
)

evaluate_task = PythonOperator(
    task_id='evaluate_model',
    python_callable=evaluate_model,
    dag=dag,
)

load_data_task >> preprocess_task >> train_model_task >> evaluate_task