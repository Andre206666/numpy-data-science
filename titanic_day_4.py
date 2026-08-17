import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_recall_curve
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
import joblib

df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv")

print(df.head())
print(df.columns)

print(df.shape)
print(df.isnull().sum())


df["age"] = df["age"].fillna(df["age"].fillna(df["age"].median()))
df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])

df = df.drop(columns=["deck", "alive", "class", "who", "embark_town", "adult_male"])

df["sex"] = df["sex"].map({"male": 0, "female": 1})

df = pd.get_dummies(df, columns=["embarked"], drop_first=True)

df["alone"] = df["alone"].astype(int)

X = df.drop(columns=["survived"])
y = df["survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

LR = LogisticRegression()
LR.fit(X_train, y_train)
print(f"LR {accuracy_score(y_test, LR.predict(X_test))}")

DT = DecisionTreeClassifier()
DT.fit(X_train, y_train)
print(f"DT {accuracy_score(y_test, DT.predict(X_test))}")

RF = RandomForestClassifier()
RF.fit(X_train, y_train)
print(f"RF {accuracy_score(y_test, RF.predict(X_test))}")

param_grid = {
    "n_estimators": [30, 40],
    "max_depth": [None, 30, 50],
    "min_samples_split": [2, 4, 6]
}

rf_base = RandomForestClassifier(class_weight="balanced", random_state=42)

grid_search = GridSearchCV(estimator=RF, param_grid=param_grid, cv=3, scoring="f1", n_jobs=-1)
grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)

print(f"Best params {grid_search.best_params_}")
print(f"Best accuracy {accuracy_score(y_test, y_pred)}")

y_scores = best_rf.predict_proba(X_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, y_pred)

plt.figure(figsize=[8, 6])
plt.plot(recall, precision, label="Precision", color="blue")
plt.plot(recall, precision, label="Recall", color="red")
plt.xlabel("Thresholds")
plt.ylabel("Precision")
plt.legend()
plt.title("Precision-Recall curve")
plt.show()

xgboost = XGBClassifier()
xgboost.fit(X_train, y_train)

predictions = xgboost.predict(X_test)

print(f"XGBoost accuracy {accuracy_score(y_test, predictions)}")

joblib.dump(best_rf, "titanic_day_4_.pkl")
print("Model saved")