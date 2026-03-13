import pickle
import os
import numpy as np
from typing import List

from data import get_target_names

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "cancer_model.pkl")

# Load model once at import time — avoids reloading on every request
_model = None


def load_model():
    """Load the trained model from disk. Raises FileNotFoundError if not trained yet."""
    global _model
    model_path = os.path.abspath(MODEL_PATH)

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run train.py first."
        )

    with open(model_path, "rb") as f:
        _model = pickle.load(f)

    return _model


def get_model():
    """Return the cached model, loading it if necessary."""
    global _model
    if _model is None:
        load_model()
    return _model


def is_model_loaded() -> bool:
    """Check whether the model file exists and is loadable."""
    try:
        get_model()
        return True
    except FileNotFoundError:
        return False


def predict(features: List[float]) -> dict:
    """
    Run inference on a single sample.

    Args:
        features: List of 30 float values matching breast cancer feature order.

    Returns:
        dict with keys: prediction (int), label (str), confidence (float), model (str)
    """
    model = get_model()
    target_names = get_target_names()

    X = np.array(features).reshape(1, -1)
    prediction = int(model.predict(X)[0])
    probabilities = model.predict_proba(X)[0]
    confidence = round(float(probabilities[prediction]), 4)
    label = target_names[prediction]

    return {
        "prediction": prediction,
        "label": label,
        "confidence": confidence,
        "model": type(model).__name__,
    }