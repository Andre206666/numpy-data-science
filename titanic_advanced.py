import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression



df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")

df["Age"] = df["Age"].fillna(df["Age"].median())
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df = df.dropna(subset=["Embarked"])

df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

X = df[["Pclass", "Sex", "Age", "Fare", "FamilySize", "IsAlone"]]
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    "n_estimators": [100, 150, 200],
    "max_depth": [3, 5, 7]
}

grid = GridSearchCV(RandomForestClassifier(), param_grid)
grid.fit(X_train, y_train)

print(f"Best params: {grid.best_params_}")
print(f"Best score: {grid.best_score_:.2f}")

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
print(f"RF {accuracy_score(y_test, rf.predict(X_test))}")


dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
print(f"DT {accuracy_score(y_test, dt.predict(X_test))}")

lr = LogisticRegression()
lr.fit(X_train, y_train)
print(f"LR {accuracy_score(y_test, lr.predict(X_test))}")
