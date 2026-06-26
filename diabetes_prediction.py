import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

np.random.seed(42)
n = 300

data = {
    "age":     np.random.randint(20, 80, n),
    "bmi":     np.random.normal(28, 6, n).clip(15, 50),
    "glucose": np.random.normal(100, 25, n).clip(60, 200),
    "pressure": np.random.normal(70, 10, n).clip(40, 100)
}

df = pd.DataFrame(data)
df["diabetes"] = ((df["age"] > 45) &
                  (df["bmi"] > 28) &
                  (df["glucose"] > 100)).astype(int)

print(df.head())
print(f"\nDiabetes cases: {df['diabetes'].sum()}")
print(f"Healthy cases: {(df['diabetes'] == 0).sum()}")

sns.heatmap(df[["age", "bmi", "glucose", "pressure", "diabetes"]].corr(), annot=True, cmap="coolwarm")
plt.title("Diabetes cases")
plt.show()

sns.scatterplot(data=df, x="bmi", y="glucose", hue="diabetes")
plt.title("bmi vs glucose")
plt.show()

X = df[["age", "bmi", "glucose", "pressure"]].values
y = df["diabetes"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, predictions)}")

print(f"Classification report: {classification_report(y_test, predictions)}")

print(model.predict([[55, 32, 140, 80]]))
print(model.predict([[25, 20, 85, 65]]))