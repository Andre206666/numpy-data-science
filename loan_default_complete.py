import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, precision_recall_curve, accuracy_score
import seaborn as sns

df = pd.read_csv("https://raw.githubusercontent.com/dsrscientist/DSData/master/loan_prediction.csv")

print("First 5 rows:\n", df.head())
print("\nMissing values before:\n", df.isnull().sum())

df["Gender"] = df["Gender"].fillna(df["Gender"].mode()[0])
df["Married"] = df["Married"].fillna(df["Married"].mode()[0])
df["Dependents"] = df["Dependents"].fillna(df["Dependents"].mode()[0])
df["Self_Employed"] = df["Self_Employed"].fillna(df["Self_Employed"].mode()[0])
df["Credit_History"] = df["Credit_History"].fillna(df["Credit_History"].mode()[0])

if "LoanAmount" in df.columns:
    df["LoanAmount"] = df["LoanAmount"].fillna(df["LoanAmount"].median())
if "Loan_Amount_Term" in df.columns:
    df["Loan_Amount_Term"] = df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].median())

X = df.drop(columns=["Loan_Status", "Loan_ID"], errors="ignore")
X = pd.get_dummies(X, drop_first=True)

y = df["Loan_Status"].map({'N': 0, 'Y': 1})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LogisticRegression(max_iter=10000)
lr.fit(X_train, y_train)
print(f"\nLR Accuracy: {accuracy_score(y_test, lr.predict(X_test)):.4f}")

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
print(f"RF Accuracy: {accuracy_score(y_test, rf.predict(X_test)):.4f}")

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
print(f"DT Accuracy: {accuracy_score(y_test, dt.predict(X_test)):.4f}")

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 10, None]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
grid_search.fit(X_train, y_train)
print(f"\nBest params: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")

print("\nTarget Distribution (%):\n", y_train.value_counts(normalize=True) * 100)

final_model = RandomForestClassifier(
    **grid_search.best_params_,
    class_weight="balanced",
    random_state=42
)
final_model.fit(X_train, y_train)
print("\nFinal Model Classification Report:\n")
print(classification_report(y_test, final_model.predict(X_test)))

importance = pd.DataFrame({
    "feature": X.columns,
    "importance": final_model.feature_importances_
}).sort_values("importance", ascending=False).head(10)

print("\nTop 10 Features:\n", importance)

plt.figure(figsize=(10, 6))
sns.barplot(data=importance, x="importance", y="feature")
plt.title("Top 10 Feature Importance - Loan Prediction")
plt.show()

y_scores = final_model.predict_proba(X_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, y_scores)

plt.figure(figsize=(8, 5))
plt.plot(thresholds, precision[:-1], label="Precision", color="blue")
plt.plot(thresholds, recall[:-1], label="Recall", color="green")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.legend()
plt.title("Precision-Recall Tradeoff - Loan Prediction")
plt.show()

X_test_copy = X_test.copy()
X_test_copy["actual"] = y_test
X_test_copy["predicted"] = final_model.predict(X_test)
X_test_copy.to_csv("loan_prediction_results.csv", index=False)
print("\nSaved predictions to loan_prediction_results.csv!")