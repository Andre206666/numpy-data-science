import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

np.random.seed(42)
n = 500

data = {
    "age": np.random.randint(20, 80, n),
    "bmi": np.random.uniform(15, 45, n),
    "glucose": np.random.randint(60, 200, n),
    "insulin": np.random.randint(10, 200, n)
}

df = pd.DataFrame(data)
df["diabetes"] = ((df["glucose"] > 140) | ((df["bmi"] > 30) & (df["age"] > 50))).astype(int)

X = df[["age", "bmi", "glucose", "insulin"]]
y = df["diabetes"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
print(f"LogisticRegression: {accuracy_score(y_test, lr.predict(X_test)):.2f}")

dt = DecisionTreeClassifier(max_depth=4)
dt.fit(X_train, y_train)
print(f"DecisionTree: {accuracy_score(y_test, dt.predict(X_test)):.2f}")

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
print(f"RandomForest: {accuracy_score(y_test, rf.predict(X_test)):.2f}")

importance = pd.DataFrame({
    "feature": ["age", "bmi", "glucose", "insulin"],
    "importance": rf.feature_importances_
}).sort_values("importance", ascending=False)
print(importance)