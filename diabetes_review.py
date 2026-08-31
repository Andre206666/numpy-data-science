from sklearn.datasets import load_diabetes
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
diabetes = load_diabetes()

X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
y = (diabetes.target > diabetes.target.mean()).astype(int)

print(X.head())
print(pd.Series(y).value_counts())
X["s_ratio"] = X["s1"] / X["s2"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

RF = RandomForestClassifier()
RF.fit(X_train, y_train)
print(f"RF {accuracy_score(y_test, RF.predict(X_test))}")

print(classification_report(y_test, RF.predict(X_test)))

param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [5, 10, 15],
    "min_samples_split": [2, 4]
}
grid = GridSearchCV(RandomForestClassifier(), param_grid)
grid.fit(X_train, y_train)
print(f"Best params: {grid.best_params_}")
print(f"Best score: {grid.best_score_}")

xgboost = XGBClassifier()
xgboost.fit(X_train, y_train)
print(f"XGBoost: {xgboost.score(X_train, y_train)}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

RF_scaled = RandomForestClassifier()
RF_scaled.fit(X_train, y_train)
print(f"RF scaled: {accuracy_score(y_test, RF_scaled.predict(X_test_scaled))}")