from sklearn.datasets import load_wine
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = pd.Series(wine.target)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

RF = RandomForestClassifier()
RF.fit(X_train, y_train)
print(f"RF score: {RF.score(X_test, y_test)}")


predictions = RF.predict(X_train)
print(f"Accuracy: {accuracy_score(y_test, RF.predict(X_test))}")

param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [None, 5, 10]
}

grid = GridSearchCV(estimator=RF, param_grid=param_grid, cv=3)
grid.fit(X_train, y_train)
print(grid.best_params_)