from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, precision_recall_curve
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
import joblib



df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv")

print(df.head())
print(df.shape)
print(df.isnull().sum())

df["bmi_category"] = df["BMI"].apply(lambda x: "Low" if x < 18.5 else "High" if x > 25 else "Normal")
df["age_group"] = df["Age"].apply(lambda x: "Young" if x < 30 else "Senior" if x > 50 else "Middle")

df["bmi_category"] = df["bmi_category"].map({"Low": 0, "Normal": 1, "High": 2})
df["age_group"] = df["age_group"].map({"Young": 0, "Middle": 1, "Senior": 2})

X = df.drop(columns=["Outcome"])
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

LR = LogisticRegression()
LR.fit(X_train, y_train)
print(f"LR {accuracy_score(y_test, LR.predict(X_test))}")

DT = DecisionTreeClassifier()
DT.fit(X_train, y_train)
print(f"DT {accuracy_score(y_test, DT.predict(X_test))}")

RF = RandomForestClassifier()
RF.fit(X_train, y_train)
print(f"RF {accuracy_score(y_test, RF.predict(X_test))}")

param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [None, 20, 30],
    "min_samples_split": [2, 4]
}
rf_base = RandomForestClassifier(class_weight="balanced", random_state=42)

grid_search = GridSearchCV(estimator=rf_base, param_grid=param_grid, cv=3, scoring="f1", n_jobs=-1)
grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)

print(f"Best params: {grid_search.best_params_}")
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))

y_scores = best_rf.predict_proba(X_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, y_scores)

plt.figure(figsize=(8, 5))
plt.plot(thresholds, precision[:-1], label="Precision", color="blue")
plt.plot(thresholds, recall[:-1], label="Recall", color="red")
plt.xlabel("Threshold")
plt.ylabel("Precision")
plt.legend()
plt.title("Precision-Recall curve")
plt.show()

xgboost = XGBClassifier(n_estimators=100, random_state=42)
xgboost.fit(X_train, y_train)

predictions = xgboost.predict(X_test)
print(f"XGBoost accuracy: {accuracy_score(y_test, predictions)}")

joblib.dump(best_rf, "diabetes_model.pkl")
print("Model saved!")

loaded_model = joblib.load("diabetes_model.pkl")

loaded_predictions = loaded_model.predict(X_test)
print(f"Loaded model accuracy: {accuracy_score(y_test, loaded_predictions)}")