import pandas as pd
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
print(df.head())
print(df.shape)
print(df.isnull().sum())

df["Age"] = df["Age"].fillna(df["Age"].median())

df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

X = df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]]
y = df["Survived"]
print(df.head())
print(df.isnull().sum())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))

print(f"Accuracy: {accuracy}")
print(classification_report(y_test, model.predict(X_test)))

importance = pd.DataFrame({
    "feature": ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"],
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)
print(importance)

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
print(f"LogisticRegression: {accuracy_score(y_test, lr.predict(X_test)):.2f}")

dt = DecisionTreeClassifier(max_depth=4)
dt.fit(X_train, y_train)
print(f"DecisionTree: {accuracy_score(y_test, dt.predict(X_test)):.2f}")

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
print(f"RandomForest: {accuracy_score(y_test, rf.predict(X_test)):.2f}")

importance = pd.DataFrame({
    "feature": ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"],
    "importance": rf.feature_importances_
}).sort_values("importance", ascending=False)

sns.barplot(data=importance, x="importance", y="feature")
plt.title("Feature Importance - Titanic")
plt.show()

