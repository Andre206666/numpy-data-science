from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = iris.target
print(df.head())
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = iris.target
print(df.head())


X = df[["sepal length (cm)", "sepal width (cm)", "petal width (cm)"]]
y = df["species"]

X_train,X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
print(f"DT: {accuracy_score(y_test, dt.predict(X_test))}")

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
print(f"RF: {accuracy_score(y_test, rf.predict(X_test))}")

lr = LogisticRegression()
lr.fit(X_train, y_train)
print(f"LR: {accuracy_score(y_test, lr.predict(X_test))}")

importance = pd.DataFrame({
    "feature": ["sepal length (cm)", "sepal width (cm)", "petal width (cm)"],
    "importance": rf.feature_importances_
})
print(importance)

print(rf.predict([[12, 23, 24]]))