from sklearn.datasets import load_breast_cancer
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report


cancer = load_breast_cancer()
X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
y = cancer.target

print(X.describe())
print(X.shape)

plt.figure(figsize=(12, 10))
sns.heatmap(X.corr(), annot=True, cmap="coolwarm", fmt=".1f")
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
print(f"\nClassification report")
print(classification_report(y_test, predictions))

importance = pd.DataFrame({
    "feature": X.columns,
    "importance": best_model.feature_importances_
}).sort_values("importance", ascending=False).head(10)

sns.barplot(data=importance, x="feature", y="importance")
plt.title("Feature importance")
plt.show()

print(best_model.predict(X_test.iloc[[0]]))

X["predicted"] = best_model.predict(X)
X.to_csv("breast_cancer_results.csv", index=False)
print("Saved!")