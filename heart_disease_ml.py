import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("https://raw.githubusercontent.com/dsrscientist/dataset1/master/heart_disease.csv")

print(df.head())
print(df.shape)
print(df.describe())
print(df.isnull().sum())

X = df[["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thal"]]
y = df[["target"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
print(f"LR: {accuracy_score(y_test, lr.predict(X_test)):.2f}")

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
print(f"RF: {accuracy_score(y_test, rf.predict(X_test)):.2f}")

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
print(f"DT: {accuracy_score(y_test, dt.predict(X_test)):.2f}")

accuracy = accuracy_score(y_test, lr.predict(X_test))
print(f"Accuracy: {accuracy}")

importance = pd.DataFrame({
    "feature": ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thal"],
    "importance": rf.feature_importances_
})
print(importance)

print(rf.predict([[45, 1, 0, 120, 200, 0, 0, 2]]))
print(rf.predict([[65, 1, 3, 160, 280, 1, 1, 3]]))

