import seaborn as sns
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = sns.load_dataset("titanic")
print(df.head())
print(df.columns)
print(df.shape)

df["age"] = df["age"].fillna(df["age"].median())
df["sex"] = df["sex"].map({"male": 0, "female": 1})

X = df[["pclass", "sex", "age", "sibsp", "parch", "fare"]]
y = df["survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

lr = LogisticRegression()
lr.fit(X_train, y_train)
print(f"lr: {accuracy_score(y_test, lr.predict(X_test))}")

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
print(f"dt: {accuracy_score(y_test, dt.predict(X_test))}")

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
print(f"rf: {accuracy_score(y_test, rf.predict(X_test))}")

importance = pd.DataFrame({
    "feature": ["pclass", "sex", "age", "sibsp", "parch", "fare"],
    "importance": rf.feature_importances_
}).sort_values("importance", ascending=False)
print(importance)