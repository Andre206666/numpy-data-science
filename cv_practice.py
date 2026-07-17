import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


wine = load_iris()
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = wine.target

for name, model in[
    ("LR", LogisticRegression(max_iter=1000)),
    ("DT", DecisionTreeClassifier()),
    ("RF", RandomForestClassifier())
]:
    scores = cross_val_score(model, X, y, cv=5)
    print(f"{name}: {scores.mean():.2f} + {scores.std():.2f}")

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 10, None]
}
grid = GridSearchCV(model, param_grid)
grid.fit(X, y)

print(f"Best params: {grid.best_params_}")
print(f"Best score: {grid.best_score_}")
