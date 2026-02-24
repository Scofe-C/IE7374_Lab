import sys
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

THRESHOLD = 0.95

# Load data and train
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Validate
accuracy = model.score(X_test, y_test)

# Print accuracy so the workflow can capture it
print(f"{accuracy:.4f}")

# Exit code drives pass/fail in GitHub Actions
if accuracy < THRESHOLD:
    sys.exit(1)