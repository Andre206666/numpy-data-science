import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("https://raw.githubusercontent.com/dsrscientist/dataset1/master/mushrooms.csv")
print(df.head())
print(df.shape)
print(df.columns)

df_encoded = pd.get_dummies(df)
print(df_encoded.head())

y = df_encoded["class_p"]

X = df_encoded.drop(["class_e", "class_p"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
    "feature": X.columns,
    "importance": rf.feature_importances_
}).sort_values("importance", ascending=False)
print(importance)

cm = confusion_matrix(y_test, dt.predict(X_test))
print(cm)

print(classification_report(y_test, dt.predict(X_test)))

scores = cross_val_score(lr, X, y, cv=5)
print(scores.mean())

one_mushroom = X_test.iloc[[0]]

print(rf.predict(one_mushroom))
print(rf.predict_proba(one_mushroom))

sns.barplot(data=importance.head(10), x="feature", y="importance")
plt.show()
