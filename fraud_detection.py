import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import RocCurveDisplay

df = pd.read_csv("https://raw.githubusercontent.com/nsethi31/Kaggle-Data-Credit-Card-Fraud-Detection/master/creditcard.csv")

df = df.sample(n=50000, random_state=42)

print(df.shape)
print(df["Class"].value_counts())
print(df["Class"].value_counts(normalize=True) * 100)
print(df.isna().sum())

X = df.drop(columns=["Class", "Time"])
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_baseline = RandomForestClassifier(class_weight="balanced", random_state=42)
rf_baseline.fit(X_train, y_train)

y_pred = rf_baseline.predict(X_test)


y_pred_balanced = rf_baseline.predict(X_test)
print("Baseline RandomForest")
print(classification_report(y_test, y_pred))

print("Random Forest with class_weigth=balanced")
print(classification_report(y_test, y_pred_balanced))

param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5]
}

rf_base = RandomForestClassifier(class_weight="balanced", random_state=42)

grid_search = GridSearchCV(
    estimator=rf_base,
    param_grid=param_grid,
    cv=3,
    scoring="f1",
    n_jobs=-1
)

print("Starting GridSearchCV this might take a minute")
grid_search.fit(X_train, y_train)

print(f"\nBest Parameters found: {grid_search.best_params_}")

best_rf = grid_search.best_estimator_
y_pred_best = best_rf.predict(X_test)
print(f"Best Model Classification Report")
print(classification_report(y_test, y_pred_best))

importances = best_rf.feature_importances_

feature_importances_df = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": importances
})

feature_importances_df = feature_importances_df.sort_values(by="Importance", ascending=False)
fig, ax = plt.subplots(figsize=(8, 6))
RocCurveDisplay.from_estimator(
    best_rf,
    X_test,
    y_test,
    name=f"Optimized RF",
    ax=ax
)

plt.plot([0, 1], [0, 1], color='navy', linestyle='--', label="Random Guess (AUC = 0.5)")
plt.title("Top 16 Feature Importances - Final Random Forest Model")
plt.xlabel("Relative Importances")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()