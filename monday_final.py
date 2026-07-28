import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")

df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

df["FamilySize"] = df["Sibsp"] + df["Parch"] + 1

X = df["Pclass", "Sex", "Age", "Fare", "FamilySize"]
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

scores = cross_val_score(model, X, y, cv=5)
print(f"Cross validation scores: {scores}")
print(f"Mean CV accuracy: {scores.mean()}")

predictions = model.predict(X_test)
print(f"Test Accuracy: {accuracy_score(y_test, predictions):.2f}")
print(classification_report(y_test, predictions))


