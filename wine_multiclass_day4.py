from sklearn.datasets import load_wine
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
import joblib



wine = load_wine()

X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = pd.Series(wine.target)

print(X.head())
print(X.shape)
print(y.value_counts())
print(wine.target_names)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

LR = LogisticRegression(max_iter=1000)
LR.fit(X_train, y_train)
print(f"LR {accuracy_score(y_test, LR.predict(X_test))}")

RF = RandomForestClassifier()
RF.fit(X_train, y_train)
print(f"RF {accuracy_score(y_test, RF.predict(X_test))}")

DT = DecisionTreeClassifier()
DT.fit(X_train, y_train)
print(f"DT {accuracy_score(y_test, DT.predict(X_test))}")

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 4, 6]
}

grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid=param_grid, cv=5)
grid.fit(X_train, y_train)

print(f"Best params: {grid.best_params_}")
print(f"Best score: {grid.best_score_:.2f}")

best_rf = grid.best_estimator_

cm = confusion_matrix(y_test, best_rf.predict(X_test))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=wine.target_names)
disp.plot(cmap="Blues")
plt.title("Wine Classification Confusion Matrix")
plt.show()

print(cm)

importance = pd.DataFrame({
    "feature": X.columns,
    "importance": best_rf.feature_importances_
}).sort_values("importance", ascending=False)

print(importance.head(10))


xgb = XGBClassifier(n_estimators=100, random_state=42)
xgb.fit(X_train, y_train)
print(f"XGBoost: {accuracy_score(y_test, xgb.predict(X_test))}")

joblib.dump(best_rf, "wine_multiclass_model.pkl")
print("Model saved!")