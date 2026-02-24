"""
Airflow DAG for Iris Classification Pipeline
DAG: iris_classification_pipeline
Tasks: load_data -> preprocess -> train_model -> evaluate_model
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
sys.path.insert(0, '/opt/airflow')
from src.ml_functions import load_data, data_preprocessing, build_save_model, evaluate_model
from airflow import configuration as conf

# Enable pickle support for XCom (allows passing data between tasks)
conf.set('core', 'enable_xcom_pickling', 'True')

# Define default arguments for the DAG
default_args = {
    'owner': 'data_scientist',
    'start_date': datetime(2026, 2, 5),
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
}

# Create the DAG
dag = DAG(
    'iris_classification_pipeline',
    default_args=default_args,
    description='ML Pipeline for Iris Classification using Random Forest',
    schedule_interval=None,  # Manual trigger only
    catchup=False,
    tags=['machine_learning', 'classification', 'iris'],
)

# Task 1: Load the Iris dataset
load_data_task = PythonOperator(
    task_id='load_data',
    python_callable=load_data,
    dag=dag,
)

# Task 2: Preprocess data (train/test split, scaling)
preprocess_task = PythonOperator(
    task_id='preprocess_data',
    python_callable=data_preprocessing,
    op_args=[load_data_task.output],
    dag=dag,
)

# Task 3: Train and save Random Forest model
train_model_task = PythonOperator(
    task_id='train_model',
    python_callable=build_save_model,
    op_args=[preprocess_task.output, 'iris_rf_model.pkl'],
    dag=dag,
)

# Task 4: Evaluate the model on test set
evaluate_task = PythonOperator(
    task_id='evaluate_model',
    python_callable=evaluate_model,
    op_args=[train_model_task.output],
    dag=dag,
)

# Define task dependencies (pipeline flow)
load_data_task >> preprocess_task >> train_model_task >> evaluate_task