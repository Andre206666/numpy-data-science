import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

np.random.seed(42)
n = 400

data = {
    "tumor_size": np.random.uniform(0.5, 5.0, 400),
    "age": np.random.randint(20, 80, 400),
    "cell_density": np.random.randint(1, 10, 400),

}

df = pd.DataFrame(data)
df["malignant"] = ((df["tumor_size"] > 3) | (df["cell_density"] > 7)).astype(int)

print(df.describe())
print(f"Mallignant: {df["malignant"].sum()}")
print(f"Healthy: {(df['malignant'] == 0).sum()}")
sns.heatmap(df[["tumor_size", "age", "cell_density"]].corr(), annot=True, cmap="YlGnBu")
plt.title("Heatmap")
plt.show()

sns.scatterplot(data=df, x="tumor_size", y="cell_density", hue="malignant")
plt.title("tumor_size vs cell_density")
plt.show()

X = df[["tumor_size", "age", "cell_density"]].values
y = df["malignant"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, predictions)}")
print(f"Classification report: {classification_report(y_test, predictions)}")
print(model.predict([[3.5, 45, 8]]))

df["risk_level"] = df.apply(
    lambda row: "High" if (row["tumor_size"] > 3.5 and row["cell_density"] > 6)
    else "Medium" if (row["tumor_size"] > 2 or row["cell_density"] > 5)
    else "Low", axis=1)
print(df["risk_level"].value_counts())

sns.barplot(data=df, x="risk_level", y="malignant")
plt.title("Risk level")
plt.show()

print(model.predict([[4.5, 48, 9]]))
print(model.predict([[2.0, 35, 4]]))
print(model.predict([[1.0, 25, 2]]))

df.to_csv("cancer_results.csv", index=False)
print("Saved!")