import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

df = pd.read_csv("https://raw.githubusercontent.com/dsrscientist/DSData/master/winequality-red.csv")
df["good_wine"] = df["quality"].apply(lambda x: 1 if x >= 7 else 0)

X = df.drop(columns=["good_wine", "quality"])
y = df["good_wine"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

xgboost = XGBClassifier(n_estimators=100, random_state=42)
xgboost.fit(X_train, y_train)

predictions = xgboost.predict(X_test)
print(f"XGBoost Accuracy: {accuracy_score(y_test, predictions):.2f}")

feature_importances = pd.DataFrame({
    "feature": X.columns,
    "importance": xgboost.feature_importances_
}).sort_values(by="importance", ascending=False)

print(feature_importances)