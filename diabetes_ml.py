import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv")
print(df.head())
print(df.isnull().sum())

X = df[["Pregnancies", "Glucose", "BloodPressure", "DiabetesPedigreeFunction", "Age"]]
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
print(f"RF: {accuracy_score(y_test, rf.predict(X_test)):.2f}")

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
print(f"LR: {accuracy_score(y_test, lr.predict(X_test)):.2f}")

accuracy = accuracy_score(y_test, lr.predict(X_test))
print(f"Accuracy: {accuracy}")


importance = pd.DataFrame({
    "feature": ["Pregnancies", "Glucose", "BloodPressure", "DiabetesPedigreeFunctions", "Age"],
    "importance": rf.feature_importances_
})
print(importance)
