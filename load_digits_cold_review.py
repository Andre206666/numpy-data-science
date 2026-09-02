from sklearn.datasets import load_digits
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression

data = load_digits()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

RF = RandomForestClassifier()
RF.fit(X_train, y_train)
print(f"Rf {accuracy_score(y_test, RF.predict(X_test))}")
print(classification_report(y_test, RF.predict(X_test)))

DT = DecisionTreeClassifier()
DT.fit(X_train, y_train)
print(f"DT {accuracy_score(y_test, DT.predict(X_test))}")
print(classification_report(y_test, DT.predict(X_test)))

LR = LogisticRegression()
LR.fit(X_train, y_train)
print(f"LR {accuracy_score(y_test, LR.predict(X_test))}")
print(classification_report(y_test, LR.predict(X_test)))

param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 4, 6, 8]

}
print(param_grid)

grid_search = GridSearchCV(estimator=RF, param_grid=param_grid, cv=3, n_jobs=-1)
grid_search.fit(X_train, y_train)
print(f"Best params {grid_search.best_params_}")
print(f"Best score {grid_search.best_score_}")