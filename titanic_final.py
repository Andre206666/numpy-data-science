import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")

df["Sex"] = df["Sex"].astype(str).str.strip().str.lower()
df["Sex"] = df["Sex"].replace({"m": "male", "f": "female", "nan": np.nan})
df['Sex'] = df['Sex'].fillna(df['Sex'].mode()[0])

df["Embarked"] = df["Embarked"].astype(str).str.strip().str.upper()
df["Embarked"] = df["Embarked"].replace({"NAN": np.nan})
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

df.loc[(df["Age"] < 0) | (df["Age"] > 120), "Age"] = np.nan
df["Age"] = df.groupby(["Pclass", "Sex"])["Age"].transform(lambda x: x.fillna(x.median()))
df["Age"] = df["Age"].fillna(df["Age"].median())

df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
df["FarePerPerson"] = df["Fare"] / df["FamilySize"]

df["Title"] = df["Name"].str.extract(r' ([A-Za-z]+)\.')
df["Title"] = df["Title"].replace(["Dr", "Rev", "Col", "Major", "Capt"], "Rare")
df["Title"] = df["Title"].map({"Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3, "Rare": 4})
df["Title"] = df["Title"].fillna(4)

df["Sex"] = df["Sex"].map({"female": 0, "male": 1})
df = pd.get_dummies(df, columns=["Embarked"], drop_first=True, dtype=int)

features = [
    'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare',
    'FamilySize', 'IsAlone', 'Title', 'FarePerPerson',
    'Embarked_Q', 'Embarked_S'
]

X = df[features]
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {"n_estimators": [50, 100, 200], "max_depth": [3, 5, 10, None]}
grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
predictions = best_model.predict(X_test)

print(f"\nAccuracy: {accuracy_score(y_test, predictions):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, predictions))

df.to_csv("titanic_complete_results.csv", index=False)
print("Saved!")