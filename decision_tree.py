from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from sklearn.tree import export_text
from sklearn.ensemble import RandomForestClassifier


np.random.seed(42)
n = 300

data = {
    "age": np.random.randint(20, 70, n),
    "income": np.random.randint(20000, 100000, n),
    "credit_score": np.random.randint(300, 850, n)
}

df = pd.DataFrame(data)
df["approved"] = (((df["credit_score"] > 650) & (df["income"] > 40000))).astype(int)

X = df[["age", "income", "credit_score"]]
y = df["approved"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, predictions):.2f}")

print(export_text(model, feature_names=["age", "income", "credit_score"]))

model2 = DecisionTreeClassifier(max_depth=3)
model2.fit(X_train, y_train)
predictions2 = model2.predict(X_test)
print(f"Depth=3 Accuracy: {accuracy_score(y_test, predictions2):.2f}")

print(model.predict([[35, 55000, 700]]))
print(model.predict([[50, 35000, 720]]))

model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)

predictions_rf = model_rf.predict(X_test)
print(f"Random Forest Accuracy: {accuracy_score(y_test, predictions_rf)}")

importance = pd.DataFrame({
    "feature": ["age", "income", "credit_score"],
    "importance": model_rf.feature_importances_
}).sort_values(by="importance", ascending=False)
print(importance)