from sklearn.datasets import load_wine
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report


wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = wine.target

print(X.describe())
print(X.isnull().sum())
print(X.head())

plt.figure(figsize=(12, 10))
sns.heatmap(X.corr(), annot=True, cmap="coolwarm", fmt=".1f")
plt.title("Wine Features Correlation")
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
print(f"RF {accuracy_score(y_test, rf.predict(X_test))}")

lr = LogisticRegression()
lr.fit(X_train, y_train)
print(f"LR {accuracy_score(y_test, lr.predict(X_test))}")

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
print(f"DT {accuracy_score(y_test, dt.predict(X_test))}")

param_grid = {"n_estimators": [20, 40, 80], "max_depth": [2, 4, 8, None] }
grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
predictions = best_model.predict(X_test)

print(f"\nAccuracy: {accuracy_score(y_test, predictions):.2f}")
print("\nClassification Report: ")
print(classification_report(y_test, predictions))

importance = pd.DataFrame({
    "feature": X.columns,
    "importance": best_model.feature_importance()
}).sort_values("importance", ascending=False)

sns.barplot(data=importance, x="feature", y="importance")
plt.title("Feature Importances")
plt.show()

print(best_model.predict(X_test.iloc[[0]]))

X["predicted_class"] = best_model.predict(X)
X.to_csv("wine_results.csv", index=False)
print("Saved!")


