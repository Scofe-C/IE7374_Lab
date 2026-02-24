"""
ML Pipeline Functions for Iris Classification
Uses Random Forest classifier with train/test split.

Each function receives `ti` (task instance) via Airflow's provide_context
and pulls upstream data from XCom explicitly.
"""

import pickle
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def load_data(**kwargs):
    """Load Iris dataset and push serialized data to XCom."""
    print("Loading Iris dataset...")

    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target

    print(f"Dataset loaded: {df.shape[0]} samples, {len(iris.feature_names)} features")
    print(f"Classes: {list(iris.target_names)}")

    data_dict = {
        'features': df[list(iris.feature_names)].values,
        'target': df['target'].values,
        'feature_names': list(iris.feature_names),
        'target_names': list(iris.target_names),
    }
    return pickle.dumps(data_dict)


def data_preprocessing(**kwargs):
    """Pull raw data from XCom, split and scale, push preprocessed data."""
    ti = kwargs['ti']
    serialized_data = ti.xcom_pull(task_ids='load_data')

    print("Preprocessing data...")
    data_dict = pickle.loads(serialized_data)

    X, y = data_dict['features'], data_dict['target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Train set: {X_train_scaled.shape[0]} samples")
    print(f"Test set: {X_test_scaled.shape[0]} samples")

    preprocessed_dict = {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'feature_names': data_dict['feature_names'],
        'target_names': data_dict['target_names'],
    }
    return pickle.dumps(preprocessed_dict)


def build_save_model(model_filename, **kwargs):
    """Pull preprocessed data from XCom, train RandomForest, save model."""
    ti = kwargs['ti']
    serialized_data = ti.xcom_pull(task_ids='preprocess_data')

    print("Training Random Forest model...")
    data_dict = pickle.loads(serialized_data)

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(data_dict['X_train'], data_dict['y_train'])
    print(f"Model trained with {model.n_estimators} trees")

    model_path = f'/opt/airflow/models/{model_filename}'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_path}")

    result_dict = {
        'model_path': model_path,
        'X_test': data_dict['X_test'],
        'y_test': data_dict['y_test'],
        'target_names': data_dict['target_names'],
    }
    return pickle.dumps(result_dict)


def evaluate_model(**kwargs):
    """Pull model info from XCom, load model, print evaluation metrics."""
    ti = kwargs['ti']
    serialized_data = ti.xcom_pull(task_ids='train_model')

    print("Evaluating model...")
    data_dict = pickle.loads(serialized_data)

    with open(data_dict['model_path'], 'rb') as f:
        model = pickle.load(f)

    y_pred = model.predict(data_dict['X_test'])
    accuracy = accuracy_score(data_dict['y_test'], y_pred)

    print(f"\n{'=' * 50}")
    print("MODEL EVALUATION RESULTS")
    print(f"{'=' * 50}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print("\nConfusion Matrix:")
    print(confusion_matrix(data_dict['y_test'], y_pred))
    print("\nClassification Report:")
    print(classification_report(data_dict['y_test'], y_pred, target_names=data_dict['target_names']))
    print(f"{'=' * 50}\n")

    return f"Accuracy: {accuracy:.4f}"