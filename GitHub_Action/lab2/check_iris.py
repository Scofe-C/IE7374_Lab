import sys
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 1. 加载数据与训练 / Load data & Train
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 2. 验证 / Validate
accuracy = model.score(X_test, y_test)
threshold = 0.95  # 设定一个高标准 / Set a high standard

# 3. 判断结果 / Judge the result
if accuracy < threshold:
    print(f"Accuracy too low: {accuracy:.2f}")
    sys.exit(1)  # 触发 GitHub Action 报错
else:
    print(f" Pass! Accuracy: {accuracy:.2f}")
    sys.exit(0)