from sklearn.datasets import load_breast_cancer
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

DT = DecisionTreeClassifier()
DT.fit(X_train, y_train)
print(f"DT {accuracy_score(y_test, DT.predict(X_test))}")
print(classification_report(y_test, DT.predict(X_test)))

RF = RandomForestClassifier()
RF.fit(X_train, y_train)
print(f"DT {accuracy_score(y_test, RF.predict(X_test))}")
print(classification_report(y_test, RF.predict(X_test)))

LR = LogisticRegression()
LR.fit(X_train, y_train)
print(f"LR {accuracy_score(y_test, LR.predict(X_test))}")
print(classification_report(y_test, LR.predict(X_test)))

xgboost = XGBClassifier()
xgboost.fit(X_train, y_train)
print(f"XGB {accuracy_score(y_test, xgboost.predict(X_test))}")
print(classification_report(y_test, xgboost.predict(X_test)))

param_grid = {
    "n_estimators": [50, 100, 50],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 4, 6]
}
print(param_grid)

grid_search = GridSearchCV(estimator=RF, param_grid=param_grid, cv=3, n_jobs=-1)
grid_search.fit(X_train, y_train)
print(f"Best params: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_}")

best_rf = grid_search.best_estimator_
test_accuracy = accuracy_score(y_test, best_rf.predict(X_test))
print(f"Test accuracy with best model: {test_accuracy:.4f}")

