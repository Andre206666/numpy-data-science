from sklearn.datasets import load_digits
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

digits = load_digits()
X_raw = pd.DataFrame(digits.data)
y_raw = pd.Series(digits.target)

X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.2, random_state=42)

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
print(f"dt {accuracy_score(y_test, dt.predict(X_test))}")

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
print(f"rf {accuracy_score(y_test, rf.predict(X_test))}")

param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [5, 10],
    "min_samples_split": [2, 4]
}

grid = GridSearchCV(RandomForestClassifier(), param_grid=param_grid)
grid.fit(X_train, y_train)
print(f"Best params: {grid.best_params_}")
print(f"Best score: {grid.best_score_}")

best_rf = grid.best_estimator_
cm = confusion_matrix(y_test, best_rf.predict(X_test))

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Digits Classification Confusion Matrix")
plt.show()