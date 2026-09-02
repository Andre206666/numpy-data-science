import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")

df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df = df.drop(columns=["Name", "Ticket", "Cabin", "PassengerId"])
df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

X = df.drop(columns=["Survived"])
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

RF = RandomForestClassifier()
RF.fit(X_train, y_train)
print(f"RF {accuracy_score(y_test, RF.predict(X_test))}")
print(classification_report(y_test, RF.predict(X_test)))

LR = LogisticRegression()
LR.fit(X_train, y_train)
print(f"LR {accuracy_score(y_test, LR.predict(X_test))}")
print(classification_report(y_test, LR.predict(X_test)))

DT = DecisionTreeClassifier()
DT.fit(X_train, y_train)
print(f"DT {accuracy_score(y_test, DT.predict(X_test))}")
print(classification_report(y_test, DT.predict(X_test)))

xgboost = XGBClassifier()
xgboost.fit(X_train, y_train)
print(f"xgboost {accuracy_score(y_test, xgboost.predict(X_test))}")

param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [2, 4, 6],
    "learning_rate": [0.1, 0.2, 0.3]
}
print(param_grid)

grid_search = GridSearchCV(estimator=xgboost, param_grid=param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)
print(f"Best params: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_}")