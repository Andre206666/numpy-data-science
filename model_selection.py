import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import joblib
import numpy as np


data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target.ravel()

print("Data loaded succesfully")
print(f"Shape of features: {X.shape}")
print(X.head())

models = [
    ("LR", LogisticRegression(max_iter=1000)),
    ("DT", DecisionTreeClassifier()),
    ("RF", RandomForestClassifier(random_state=42))
]

print("--- Cross Validation Scores ---")
for name, model in models:
    scores = cross_val_score(model, X, y, cv=5)
    print(f"{name}: {scores.mean():.2f} (+/-{scores.std():.2f}")

print("\n--- Tuning Random Forest --- ")
rf_model = RandomForestClassifier(random_state=42)

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 10, None]
}

grid = GridSearchCV(rf_model, param_grid, cv=5)
grid.fit(X, y)
print(f"Best params: {grid.best_params_}")
print(f"Best score: {grid.best_score_}")

print("\n--- Feature Importance ---")
best_model = grid.best_estimator_
importances = best_model.feature_importances_

for name, importance in zip(data.feature_names, importances):
    print(f"{name}: {importance:.4f}")

print("\n--- Predict a new flower ---")

new_flower = [[5, 3.5, 1.3, 0,2]]


print("\n--- Predict a New Flower ---")
new_flower = np.array([[5.0, 3.5, 1.3, 0.2]])

prediction = grid.best_estimator_.predict(new_flower)
species = data.target_names[prediction[0]]

print(f"Measurements: {new_flower[0]}")
print(f"Model predicts this is a: {species}")


filename = 'best_rf_model.pkl'
joblib.dump(grid.best_estimator_, filename)
print(f"\n--- Success ---")
print(f"Model saved as '{filename}'")