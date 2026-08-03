import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, precision_recall_curve
import warnings

warnings.filterwarnings("ignore")

url = "https://raw.githubusercontent.com/nelson-wu/employee-attrition-ml/master/WA_Fn-UseC_-HR-Employee-Attrition.csv"
df = pd.read_csv(url)

df["Attrition"] = df["Attrition"].map({"No": 0, "Yes": 1})
df = df.drop(columns=["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"])

df_encoded = pd.get_dummies(df, drop_first=True)
X = df_encoded.drop(columns=["Attrition"])
y = df_encoded["Attrition"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = X_train.copy()
X_test = X_test.copy()

print("\n--- Training Class Distribution Percentages ---")
print(y_train.value_counts(normalize=True) * 100)
print("-" * 40)

lr = LogisticRegression(max_iter=10000)
lr.fit(X_train, y_train)

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

print("\nStarting GridSearchCV for Random Forest...")
param_grid = {"n_estimators": [50, 100], "max_depth": [5, 10]}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid=param_grid, cv=3)
grid_search.fit(X_train, y_train)

final_rf = RandomForestClassifier(**grid_search.best_params_, class_weight="balanced", random_state=42)
final_rf.fit(X_train, y_train)

dt_tuned = DecisionTreeClassifier(max_depth=5, random_state=42)
voting_model = VotingClassifier(estimators=[("lr", lr), ("dt", dt_tuned), ("rf", final_rf)], voting="soft")
voting_model.fit(X_train, y_train)

print(f"\nVoting Classifier Accuracy: {accuracy_score(y_test, voting_model.predict(X_test)):.4f}")

# 7. Extract Feature Importances
importance = pd.DataFrame({
    "feature": X.columns,
    "importance": final_rf.feature_importances_
}).sort_values(by="importance", ascending=False)

print("\nTop 10 Features Driving Attrition:")
print(importance.head(10))

print("\nClassification Report:")
print(classification_report(y_test, voting_model.predict(X_test)))

y_scores = voting_model.predict_proba(X_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, y_scores)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color="blue", linewidth=2)
plt.title("Precision-Recall Curve - HR Attrition Voting Classifier")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.grid(True)
plt.show()