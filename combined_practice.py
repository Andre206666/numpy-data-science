import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv")

print(df["Outcome"].value_counts())

df["risk_level"] = df["Glucose"].apply(lambda x: "High" if x > 140 else "Normal")

print(df[["Glucose", "risk_level"]])

X = df.drop(["Outcome", "risk_level"], axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
print(f"RF {accuracy_score(y_test, rf.predict(X_test))}")

scores = cross_val_score(rf, X, y, cv=5)
print(f"Mean CV score: {scores.mean():.2f}")

predictions = rf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, predictions):.2f}")
print(classification_report(y_test, predictions))


