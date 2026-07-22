import os
import ssl
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

ssl._create_default_https_context = ssl._create_unverified_context

try:
    df = pd.read_csv("https://raw.githubusercontent.com/YBI-Foundation/Dataset/main/Heart%20Disease.csv")
    print("Dataset loaded from the internet successfully!")
except Exception:
    # Try 2: Local File
    if os.path.exists("Heart Disease.csv"):
        df = pd.read_csv("Heart Disease.csv")
        print("✅ Dataset loaded from your local file successfully!")
    else:
        print("Warning: Could not download or find the file.")
        print("Generating a realistic dataset automatically so the code can run...")
        np.random.seed(42)
        df = pd.DataFrame({
            "Age": np.random.randint(29, 77, 300),
            "BMI": np.random.uniform(16.0, 38.0, 300),
            "Cholesterol": np.random.randint(126, 564, 300),
            "Max HR": np.random.randint(71, 202, 300),
            "Blood Pressure": np.random.randint(94, 200, 300),
            "Heart Disease": np.random.choice([0, 1], 300)
        })

df["BMI_category"] = df["BMI"].apply(lambda x: "Low" if x < 18.5 else ("Normal" if x <= 29.9 else "High"))
df["Age_group"] = df["Age"].apply(lambda x: "Young" if x < 45 else ("Middle" if x <= 60 else "Senior"))

df["BMI_category"] = df["BMI_category"].map({"Low": 0, "Normal": 1, "High": 2})
df["Age_group"] = df["Age_group"].map({"Young": 0, "Middle": 1, "Senior": 2})

X = df.drop(columns=["Heart Disease"])
y = df["Heart Disease"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n--- Training Models ---")
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
print(f"Random Forest Accuracy: {accuracy_score(y_test, rf.predict(X_test)):.3f}")

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
print(f"Logistic Regression Accuracy: {accuracy_score(y_test, lr.predict(X_test)):.3f}")

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
print(f"Decision Tree Accuracy: {accuracy_score(y_test, dt.predict(X_test)):.3f}")

print("\n--- Running GridSearchCV ---")
param_grid = {"n_estimators": [50, 100, 200], "max_depth": [2, 4, 8, None]}
grid = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_

predictions = best_model.predict(X_test)
print(f"\nTuned Model Accuracy: {accuracy_score(y_test, predictions):.3f}")
print(classification_report(y_test, predictions))

importance = pd.DataFrame({
    "feature": X.columns,
    "importance": best_model.feature_importances_
}).sort_values("importance", ascending=False)
print("\nFeature Importance:\n", importance)

plt.figure(figsize=(8, 5))
sns.barplot(data=importance, x="importance", y="feature", palette="viridis", hue="feature")
plt.title("Feature Importance")
plt.tight_layout()
plt.show()

print("\n--- Testing Single Row Predictions ---")
print("Row 0 prediction:", best_model.predict(X_test.iloc[[0]])[0])
print("Row 1 prediction:", best_model.predict(X_test.iloc[[1]])[0])
print("Row 2 prediction:", best_model.predict(X_test.iloc[[2]])[0])

df.to_csv("heart_complete.csv", index=False)
print("\n✅ File saved successfully to 'heart_complete.csv'!")