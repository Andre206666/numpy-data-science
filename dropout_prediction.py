import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

np.random.seed(42)
n = 500

data = {
    "gpa":          np.random.normal(2.5, 0.8, n).clip(0, 4),
    "attendance":   np.random.randint(40, 100, n),
    "hours_study":  np.random.randint(1, 10, n),
    "failed_courses": np.random.randint(0, 5, n),
    "financial_aid": np.random.randint(0, 2, n)
}

df = pd.DataFrame(data)
df["dropout"] = ((df["gpa"] < 2.0) |
                 (df["attendance"] < 60) |
                 (df["failed_courses"] > 2)).astype(int)

print(df.describe())
print(f"\nDropouts: {df['dropout'].sum()}")
print(f"Non-dropouts: {(df['dropout'] == 0).sum()}")

sns.heatmap(df[["gpa", "attendance", "hours_study", "failed_courses", "financial_aid", "dropout"]].corr(),
            annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

sns.scatterplot(data=df, x="gpa", y="attendance", hue="dropout")
plt.title("gpi vs attendance")
plt.show()

X = df[["gpa", "attendance", "hours_study", "failed_courses", "financial_aid"]]
y = df["dropout"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print(f"Accuracy report: {accuracy_score(y_test, predictions)}")

print(f"Classification report {classification_report(y_test, predictions)}")


model.predict(pd.DataFrame([[3.5, 90, 8, 0, 1]],
              columns=["gpa", "attendance", "hours_study", "failed_courses", "financial_aid"]))