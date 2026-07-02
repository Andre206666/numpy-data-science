import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


np.random.seed(42)
n = 400

data = {
    "income": np.random.randint(20000, 100000, 400),
    "credit_score": np.random.randint(300, 850, 400),
    "loan_amount": np.random.randint(5000, 50000, 400),
    "employed": np.random.randint(0, 2, 400)

}
df = pd.DataFrame(data)
df["approved"] = (((df["credit_score"] > 650) & (df["income"] > 40000)) | (df["employed"] == 1)).astype(int)

print(df.describe())
print(f"Approved: {df['approved'].sum()}")
print(f"Rejected: {(df['approved'] == 0).sum()}")

sns.heatmap(df[["income", "credit_score", "loan_amount", "employed"]].corr(), annot=True)  # ✅
plt.title("Heatmap")
plt.show()

sns.scatterplot(data=df, x="credit_score", y="income", hue="approved")
plt.title("credit_score vs income")
plt.show()

X = df[["income", "credit_score", "loan_amount", "employed"]]
y = df["approved"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, predictions)}")
print(classification_report(y_test, predictions))

df["risk_level"] = df.apply(
    lambda row: "High" if (row["credit_score"] > 60 and row["income"] > 240)
    else "Medium" if (row["credit_score"] > 45 or row["income"] > 200)
    else "Low", axis=1)
print(model.predict([[40, 220, 120, 1]]))
print(model.predict([[42, 218, 225, 1]]))

df.to_csv("loan_approval_results.csv", index=False)
print("Saved!")