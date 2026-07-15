import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report

iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

for name, model in [("LR", LogisticRegression(max_iter=100)),
                    ("DT", DecisionTreeClassifier()),
                    ("RF", RandomForestClassifier())]:
    scores = cross_val_score(model, X, y, cv=5)
    print(f"{name}: {scores.mean():.2f} + {scores.std():.2f}")

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 10, None]
}

grid_search = GridSearchCV(model, param_grid)
grid_search.fit(X, y)

print(f"Best params: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_}")

df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")

df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
df["AgeGroup"] = pd.cut(df["Age"], bins=[0, 12, 18, 60, 100], labels=[0, 1, 2, 3])

X = df[["Pclass", "Sex", "Age", "Fare", "FamilySize", "IsAlone", "AgeGroup"]]
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
print(f"Titanic with feature engineering: {accuracy_score(y_test, rf.predict(X_test))}")

