import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

np.random.seed(42)
n = 150

data = {
    "age":          np.random.randint(22, 60, n),
    "salary":       np.random.normal(60000, 20000, n).clip(25000, 120000),
    "experience":   np.random.randint(1, 35, n),
    "satisfaction": np.random.randint(1, 10, n),
    "department":   np.random.choice(["Engineering", "Marketing", "HR", "Sales"], n)
}

df = pd.DataFrame(data)

print(df.describe())

print(df.groupby("department")["salary"].mean())
print(df.groupby("age")["salary"].mean())
sns.heatmap(df[["age", "salary", "experience", "satisfaction"]].corr(), annot=True, cmap="coolwarm")
plt.title("Age vs Salary")
plt.show()

sns.scatterplot(data=df, x="experience", y="salary", hue="department")
plt.title("Experience vs Salary")
plt.show()

df["level"] = df["experience"].apply(lambda x: "Senior" if x > 10 else "Junior")
print(df)

sns.barplot(data=df, x="department", y="satisfaction")
plt.title("Satisfaction by Department")
plt.show()
