from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def get_data():
    """
    Load the Breast Cancer Wisconsin dataset and return a train/test split.

    Returns:
        X_train, X_test, y_train, y_test as numpy arrays.
        Feature names and target names are accessible via the dataset object.
    """
    dataset = load_breast_cancer()
    X, y = dataset.data, dataset.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test


def get_feature_names():
    """Return the list of 30 feature names from the dataset."""
    return load_breast_cancer().feature_names.tolist()


def get_target_names():
    """Return class label strings: ['malignant', 'benign']"""
    return load_breast_cancer().target_names.tolist()