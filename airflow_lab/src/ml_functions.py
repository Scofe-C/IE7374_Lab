"""
ML Pipeline Functions for Iris Classification
Uses Random Forest classifier with train/test split
"""

import pickle
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def load_data():
    """
    Load Iris dataset and return as serialized data
    Returns: Serialized dictionary with features and target
    """
    print("Loading Iris dataset...")

    # Load iris dataset
    iris = load_iris()

    # Create DataFrame for better handling
    df = pd.DataFrame(
        data=iris.data,
        columns=iris.feature_names
    )
    df['target'] = iris.target
    df['target_name'] = df['target'].map({i: name for i, name in enumerate(iris.target_names)})

    print(f"Dataset loaded: {df.shape[0]} samples, {df.shape[1] - 2} features")
    print(f"Classes: {iris.target_names}")

    # Return as serialized dictionary
    data_dict = {
        'features': df.drop(['target', 'target_name'], axis=1).values,
        'target': df['target'].values,
        'feature_names': iris.feature_names,
        'target_names': iris.target_names
    }

    return pickle.dumps(data_dict)


def data_preprocessing(serialized_data):
    """
    Preprocess data: train/test split and feature scaling
    Args:
        serialized_data: Pickled data from load_data
    Returns: Serialized preprocessed data dictionary
    """
    print("Preprocessing data...")

    # Deserialize input data
    data_dict = pickle.loads(serialized_data)

    X = data_dict['features']
    y = data_dict['target']

    # Split into train and test sets (80-20 split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features (important for many ML algorithms)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Train set: {X_train_scaled.shape[0]} samples")
    print(f"Test set: {X_test_scaled.shape[0]} samples")

    # Return preprocessed data
    preprocessed_dict = {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'feature_names': data_dict['feature_names'],
        'target_names': data_dict['target_names']
    }

    return pickle.dumps(preprocessed_dict)


def build_save_model(serialized_data, model_filename):
    """
    Build Random Forest model and save to file
    Args:
        serialized_data: Pickled preprocessed data
        model_filename: Path to save the model
    Returns: Serialized model info
    """
    print("Training Random Forest model...")

    # Deserialize input data
    data_dict = pickle.loads(serialized_data)

    X_train = data_dict['X_train']
    y_train = data_dict['y_train']

    # Create and train Random Forest classifier
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    print(f"Model trained with {model.n_estimators} trees")

    # Save model to file
    model_path = f'/opt/airflow/models/{model_filename}'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    print(f"Model saved to {model_path}")

    # Return model info and test data for evaluation
    result_dict = {
        'model_path': model_path,
        'X_test': data_dict['X_test'],
        'y_test': data_dict['y_test'],
        'target_names': data_dict['target_names']
    }

    return pickle.dumps(result_dict)


def evaluate_model(serialized_data):
    """
    Load model and evaluate on test set
    Args:
        serialized_data: Pickled model info and test data
    Returns: Evaluation metrics as string
    """
    print("Evaluating model...")

    # Deserialize input data
    data_dict = pickle.loads(serialized_data)

    model_path = data_dict['model_path']
    X_test = data_dict['X_test']
    y_test = data_dict['y_test']
    target_names = data_dict['target_names']

    # Load the saved model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n{'=' * 50}")
    print(f"MODEL EVALUATION RESULTS")
    print(f"{'=' * 50}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    print(f"{'=' * 50}\n")

    return f"Model evaluation complete. Accuracy: {accuracy:.4f}"