import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv")
print(df.head())
print(df.describe())
print(df.isnull().sum())

sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

df["BMI_category"] = df["BMI"].apply(lambda x: "Low" if x < 18.5 else "High" if x > 18.5 < 30 else "Normal")
df["Age_group"] = df["Age"].apply(lambda x: "Young" if x < 30 else "Senior" if x > 30 < 50 else "Middle")

df["BMI_category"] = df["BMI_category"].map({"Low": 0, "Normal": 1, "High": 2})
df["Age_group"] = df["Age_group"].map({"Young": 0, "Middle": 1, "Senior": 2})

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train , X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
print(f"RF: {accuracy_score(y_test, rf.predict(X_test))}")

lr = LogisticRegression()
lr.fit(X_train, y_train)
print(f"LR {accuracy_score(y_test, lr.predict(X_test))}")

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
print(f"DT: {accuracy_score(y_test, dt.predict(X_test))}")

param_grid = {"n_estimators": [50, 100, 200], "max_depth": [3, 5, 10, None]}
grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
best_model.fit(X_train, y_train)

predictions = best_model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, predictions):.2f}")
print(classification_report(y_test, predictions))

importance = pd.DataFrame({
    "feature": X.columns,
    "importance": best_model.feature_importances_
}).sort_values("importance", ascending=False)

sns.barplot(data=importance, x="importance", y="feature")
plt.title("Feature Importance")
plt.show()

print(best_model.predict(X_test.iloc[[0]]))
print(best_model.predict(X_test.iloc[[1]]))
print(best_model.predict(X_test.iloc[[2]]))

df.to_csv("diabetes_results.csv", index=False)
print("Saved!")