import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, precision_recall_curve, confusion_matrix, roc_curve, auc
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import seaborn as sns

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
df = pd.read_csv(url, sep=";")

df["bound_sulfur_ratio"] = df["total sulfur dioxide"] / (
    df["free sulfur dioxide"] + 1e-5
)
df["alcohol_acidity_ratio"] = df["alcohol"] / (df["volatile acidity"] + 1e-5)

df["good_wine"] = (df["quality"] >= 7).astype(int)

y = df["good_wine"]
X = df.drop(columns=["quality", "good_wine"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

RF = RandomForestClassifier(random_state=42)
RF.fit(X_train, y_train)
print(f"RF Accuracy: {accuracy_score(y_test, RF.predict(X_test)):.4f}")

LR = LogisticRegression(max_iter=1000, random_state=42)
LR.fit(X_train, y_train)
print(f"LR Accuracy: {accuracy_score(y_test, LR.predict(X_test)):.4f}")

DT = DecisionTreeClassifier(random_state=42)
DT.fit(X_train, y_train)
print(f"DT Accuracy: {accuracy_score(y_test, DT.predict(X_test)):.4f}")

param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [2, 5, 10],
    "min_samples_split": [2, 5],
}

rf_base = RandomForestClassifier(class_weight="balanced", random_state=42)
grid_search = GridSearchCV(
    estimator=rf_base, param_grid=param_grid, cv=3, scoring="f1", n_jobs=-1
)
grid_search.fit(X_train, y_train)

print(f"Best params: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.4f}")
print(classification_report(y_test, grid_search.predict(X_test)))

y_scores = grid_search.predict_proba(X_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, y_scores)

plt.figure(figsize=(8, 6))
plt.plot(thresholds, precision[:-1], label="Precision", color="blue")
plt.plot(thresholds, recall[:-1], label="Recall", color="red")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.legend()
plt.title("Precision-Recall Curve")
plt.show()

fpr, tpr, _ = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(
    fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})"
)
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.show()

xgboost = XGBClassifier(n_estimators=100, random_state=42)
xgboost.fit(X_train, y_train)

joblib.dump(xgboost, "xgboost.pkl")
print("Model saved successfully!")

loaded_model = joblib.load("xgboost.pkl")
loaded_predictions = loaded_model.predict(X_test)
print(f"Loaded model accuracy: {accuracy_score(y_test, loaded_predictions):.4f}")

cm = confusion_matrix(y_test, loaded_predictions)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("XGBoost Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

