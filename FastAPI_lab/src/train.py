import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

from data import get_data, get_target_names

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "cancer_model.pkl")


def train():
    print("Loading data...")
    X_train, X_test, y_train, y_test = get_data()
    target_names = get_target_names()

    print("Training RandomForestClassifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))

    # Persist model
    model_path = os.path.abspath(MODEL_PATH)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print(f"\nModel saved to: {model_path}")


if __name__ == "__main__":
    train()