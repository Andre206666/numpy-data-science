import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_curve,
    fbeta_score,
    make_scorer
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv")

df["FamilySize"] = df["sibsp"] + df["parch"] + 1
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

df["age"] = df["age"].fillna(df["age"].median())
df["fare"] = df["fare"].fillna(df["fare"].median())

df = df.drop(columns=["alive", "deck", "embark_town", "who"])
df = pd.get_dummies(df, columns=["sex", "embarked", "class"], drop_first=True)

X = df.drop(columns=["survived"])
y = df["survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

LR = LogisticRegression(max_iter=1000)
LR.fit(X_train, y_train)
print(f"LR Accuracy: {accuracy_score(y_test, LR.predict(X_test)):.4f}")

DT = DecisionTreeClassifier(random_state=42)
DT.fit(X_train, y_train)
print(f"DT Accuracy: {accuracy_score(y_test, DT.predict(X_test)):.4f}")

RF = RandomForestClassifier(random_state=42)
RF.fit(X_train, y_train)
print(f"RF Accuracy: {accuracy_score(y_test, RF.predict(X_test)):.4f}")

param_grid = {
    "n_estimators": [50, 120],
    "max_depth": [None, 30, 50],
    "min_samples_split": [3, 5, 8]
}

rf_base = RandomForestClassifier(class_weight="balanced", random_state=42)
f2_scorer = make_scorer(fbeta_score, beta=2)

grid_search = GridSearchCV(estimator=rf_base, param_grid=param_grid, cv=3, scoring=f2_scorer, n_jobs=-1)
grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)

print(f"\nBest Params: {grid_search.best_params_}")
print(f"Best RF Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))

y_scores = best_rf.predict_proba(X_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, y_scores)

plt.figure(figsize=[8, 6])
plt.plot(thresholds, precision[:-1], label="Precision", color="blue")
plt.plot(thresholds, recall[:-1], label="Recall", color="red")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.legend()
plt.title("Precision & Recall vs. Decision Threshold")
plt.grid(True)
plt.show()

xgboost = XGBClassifier(random_state=42)
xgboost.fit(X_train, y_train)
print(f"XGBoost Accuracy: {accuracy_score(y_test, xgboost.predict(X_test)):.4f}")

clf1 = LogisticRegression(max_iter=1000)
clf2 = RandomForestClassifier(random_state=42)
clf3 = SVC(probability=True, random_state=42)

ensemble = VotingClassifier(
    estimators=[("lr", clf1), ("rf", clf2), ("svc", clf3)],
    voting="soft"
)

ensemble.fit(X_train, y_train)
predictions = ensemble.predict(X_test)
print(f"Ensemble Accuracy: {accuracy_score(y_test, predictions):.4f}")

joblib.dump(ensemble, "ensemble.joblib")
print("Model saved successfully!")