import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns

np.random.seed(42)
n = 500

data = {
    "salary": np.random.randint(30000, 120000, 500),
    "satisfaction": np.random.randint(1, 10, 500),
    "years": np.random.randint(1, 20, 500),
    "overtime": np.random.randint(0, 2, 500)
}
df = pd.DataFrame(data)
df["churn"] = (((df["satisfaction"] < 4) | (df["salary"] < 45000)) & (df["years"] < 5)).astype(int)

print(df.describe())
print(f"Churn: {df['churn'].sum()}")
print(f"Stayed: {(df['churn'] == 0).sum()}")

sns.heatmap(df[["salary", "satisfaction", "years", "overtime"]].corr(), annot=True, cmap="YlGnBu")
plt.title("Heatmap")
plt.show()

sns.scatterplot(data=df, x="satisfaction", y="salary", hue="churn")
plt.title("satisfaction vs salary")
plt.show()

X = df[["salary", "satisfaction", "years", "overtime"]]
y = df["churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print(f"Accuracy {accuracy_score(y_test, predictions)}")
print(classification_report(y_test, predictions))


df["risk_level"] = df.apply(
    lambda row: "High" if (row["salary"] > 100000 and row["satisfaction"] > 10)
    else "Medium" if (row["salary"] > 85000 or row["satisfaction"] > 7)
    else "Low", axis=1)

print(model.predict([[55, 32, 140, 80]]))
print(model.predict([[54, 31, 135, 75]]))

df.to_csv("employee_churn.csv", index=False)
print("Saved!")