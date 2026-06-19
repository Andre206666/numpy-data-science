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

df = pd.DataFrame(data)
df = df.clip(0, 100)  # keep scores between 0-100
print(df.head())
print(df.describe())

df["average"] = df[["math", "science", "english"]].mean(axis=1)

df["passed"] = df["average"].apply(lambda x: "Yes" if x > 70 else "No")

print("Students passed:", (df["passed"] == "Yes").sum())

sns.heatmap(df[["math", "science", "english", "hours_studied", "attendance", "average"]].corr(),
            annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

sns.scatterplot(data=df, x="hours_studied", y="average", hue="passed")
plt.title("Hours Studied vs Average")
plt.show()

sns.histplot(data=df, x="average",  hue="passed", bins=20)
plt.title("Distribution of average")
plt.show()

sns.barplot(data=df, x="passed", y="average")
plt.title("Average score by Pass/Fail")
plt.show()