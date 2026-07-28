import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier

url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
df = pd.read_csv(url)

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("--- Baseline Model Cross-Validation ---")
for name, model in [
    ("LR", LogisticRegression(max_iter=1000)),
    ("DT", DecisionTreeClassifier(random_state=42)),
    ("RF", RandomForestClassifier(random_state=42)),
]:
    scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"{name}: {scores.mean():.2f} ± {scores.std():.2f}")

print("\n--- Tuning Random Forest Hyperparameters ---")
param_grid = {"n_estimators": [50, 100, 200], "max_depth": [3, 5, None]}

grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
grid.fit(X_train, y_train)

print(f"Best Params: {grid.best_params_}")

best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

print(f"\nFinal Test Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))