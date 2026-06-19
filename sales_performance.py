import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

np.random.seed(42)
n_students = 100
data = {
    "math":    np.random.normal(70, 15, n_students),
    "science": np.random.normal(65, 12, n_students),
    "english": np.random.normal(75, 10, n_students),
    "hours_studied": np.random.randint(1, 10, n_students),
    "attendance": np.random.randint(60, 100, n_students)
}
df = pd.DataFrame(data).clip(0, 100)

# 2. Describe() — full stats summary
print(df.describe())

df["average"] = df[["math", "science", "english"]].mean(axis=1)
df["performance"] = np.where(df["average"] > 75, "High", "Low")

print("\nPerformance Counts:")
print(df["performance"].value_counts())

print("\nAverage grades by performance:")
print(df.groupby("performance")[["math", "science", "english"]].mean())

plt.figure(figsize=(10, 6))
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

sns.scatterplot(data=df, x="hours_studied", y="average", hue="performance")
plt.title("Hours Studied vs Average")
plt.show()

sns.barplot(data=df, x="performance", y="average")
plt.title("Average Score by Performance Tier")
plt.show()