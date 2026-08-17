import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.datasets import load_breast_cancer
import joblib

cancer = load_breast_cancer()
X_raw = pd.DataFrame(cancer.data, columns=cancer.feature_names)
y_raw = pd.Series(cancer.target)

print(X_raw.head())
print(X_raw.shape)
print(y_raw.value_counts())

X_raw["radius_texture_ratio"] = X_raw["mean radius"] / X_raw["mean texture"]
X_raw["area_perimeter_ratio"] = X_raw["mean area"] / X_raw["mean perimeter"]

X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.2, random_state=42)

LR = LogisticRegression()
LR.fit(X_train, y_train)
print(f"LR {accuracy_score(y_test, LR.predict(X_test))}")

RF = RandomForestClassifier()
RF.fit(X_train, y_train)
print(f"RF {accuracy_score(y_test, RF.predict(X_test))}")

DT = DecisionTreeClassifier()
DT.fit(X_train, y_train)
print(f"DT {accuracy_score(y_test, DT.predict(X_test))}")


param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [None, 40, 50],
    "min_samples_split": [2, 4]
}

rf_base = RandomForestClassifier(class_weight="balanced", random_state=42)

grid_search = GridSearchCV(estimator=rf_base, param_grid=param_grid, cv=3, scoring="f1", n_jobs=-1)
grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)

print(f"Best params {grid_search.best_params_}")
print(f"Best accuracy {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))

y_scores = best_rf.predict_proba(X_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, y_scores)

plt.figure(figsize=[8, 4])
plt.plot(recall, precision, label="Precision", color="blue")
plt.plot(recall, precision, label="Recall", color="red")
plt.xlabel("Thresholds")
plt.ylabel("Precision")
plt.legend()
plt.title("Precision-Recall curve")
plt.show()

xgboost = XGBClassifier()
xgboost.fit(X_train, y_train)

predictions = xgboost.predict(X_test)
print(f"XGBoost acuuracy {accuracy_score(y_test, predictions)}")

joblib.dump(best_rf, "breast_cancer_day4.pkl")
print("Model saved!")

print("End")