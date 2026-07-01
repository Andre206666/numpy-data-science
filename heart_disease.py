import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns

np.random.seed(42)

data = {
    "age": np.random.randint(30, 80, 350),
    "cholesterol": np.random.randint(150, 300, 350),
    "blood_pressure": np.random.randint(80, 180, 350),
    "smoker": np.random.randint(0, 2, 350)
}

df = pd.DataFrame(data)
df["heart_disease"] = (((df["age"] > 55) & (df["cholesterol"] > 220)) | (df["smoker"] == 1)).astype(int)

print(df.describe())
print(f"Heart disease: {df['heart_disease'].sum()}")
print(f"Healthy: {(df['heart_disease'] == 0).sum()}")

sns.heatmap(df[["age", "cholesterol", "blood_pressure", "smoker"]].corr(), annot=True, cmap="YlGnBu")
plt.title("Heatmap")
plt.show()

sns.scatterplot(data=df, x="age", y="cholesterol", hue="heart_disease")
plt.title("Age vs Cholesterol")
plt.show()

X = df[["age", "cholesterol", "blood_pressure", "smoker"]]
y = df["heart_disease"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LogisticRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, predictions):.2f}")
print(classification_report(y_test, predictions))

df["risk_level"] = df.apply(
    lambda row: "High" if (row["age"] > 60 and row["cholesterol"] > 240)
    else "Medium" if (row["age"] > 45 or row["cholesterol"] > 200)
    else "Low", axis=1)

print(model.predict([[65, 250, 130, 1]]))
print(model.predict([[35, 160, 90, 0]]))

df.to_csv("heart_disease_results.csv", index=False)
print("Saved!")