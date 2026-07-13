import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

df = sns.load_dataset("penguins")
print(df.head())
print(df.isnull().sum())

df = df.dropna()
print(df.dtypes)
print(df["sex"].unique())
print(df["species"].unique())

df["species"] = df["species"].map({"Adelie": 0, "Chinstrap": 1, "Gentoo": 2})

df["sex"] = df["sex"].map({"Female": 0, "Male": 1})

X = df[["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g", "sex"]]
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
print(f"LR: {accuracy_score(y_test, lr.predict(X_test)):.2f}")

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
print(f"DT {accuracy_score(y_test, dt.predict(X_test)):.2f}")

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
print(f"Random Forest: {accuracy_score(y_test, rf.predict(X_test)):.2f}")

importance = pd.DataFrame({
    "feature": ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g", "sex"],
    "importance": rf.feature_importances_
})
print(importance)


