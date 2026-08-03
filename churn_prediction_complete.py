import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, precision_recall_curve

df = pd.read_csv("https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv")

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

df_encoded = pd.get_dummies(df.drop("customerID", axis=1), drop_first=True)

X = df_encoded.drop(columns=["Churn"])
y = df_encoded["Churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = X_train.copy()
X_test = X_test.copy()


print("--- Training Class Distribution Percentages ---")
print(y_train.value_counts(normalize=True)* 100)
print("-" * 40)


train_mean = X_train["TotalCharges"].mean()
X_train["TotalCharges"] = X_train["TotalCharges"].fillna(train_mean)
X_test["TotalCharges"] = X_test["TotalCharges"].fillna(train_mean)

lr = LogisticRegression(max_iter=10000)
lr.fit(X_train, y_train)
print(f"LR: {accuracy_score(y_test, lr.predict(X_test)):.4f}")

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
print(f"DT: {accuracy_score(y_test, dt.predict(X_test)):.4f}")

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
print(f"RF: {accuracy_score(y_test, rf.predict(X_test)):.4f}")

best_model = RandomForestClassifier(random_state=42)

print("Starting GridSearchCV...")

param_grid = {"n_estimators": [50, 100], "max_depth": [5, 10]}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid=param_grid, cv=3)
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_}")

final_model = RandomForestClassifier(**grid_search.best_params_, class_weight="balanced", random_state=42)
final_model.fit(X_train, y_train)

dt_tuned = DecisionTreeClassifier(max_depth=5, random_state=42)
voting_model = VotingClassifier(estimators=[("lr", lr), ("dt", dt_tuned), ("rf", final_model)], voting="soft")
voting_model.fit(X_train, y_train)

print(f"\nVoting accuracy: {accuracy_score(y_test, voting_model.predict(X_test)):.4f}")

importance = pd.DataFrame({
    "feature": X.columns,
    "importance": final_model.feature_importances_
}).sort_values(by="importance", ascending=False)

print("\nTop 10 Features:")
print(importance.head(10))

print("\nClassification Report:")
print(classification_report(y_test, voting_model.predict(X_test)))

y_scores = voting_model.predict_proba(X_test)[:, 1]
print(f"\nSuccesfully calculated y_scores for the Precission-Recall curve:")

precision, recall, thresholds = precision_recall_curve(y_test, y_scores)

plt.figure(figsize=[6, 8])
plt.plot(recall, precision, color="blue", linewidth=2)
plt.title("Precision-Recall curve - Voting Classifier")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.grid(True)
plt.show()