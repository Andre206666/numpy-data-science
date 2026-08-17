import pandas as pd
import seaborn as sns

df = sns.load_dataset("tips")

print(df[df["tip"] > 5])

print(df.groupby("day")["total_bill"].mean())

df["big_tip"] = df["tip"].apply(lambda x: "Yes" if x > 4 else "No")
print(df[["tip", "big_tip"]])

print(df.sort_values(by="total_bill", ascending=False))

print(df[(df["day"] == "Sat") & (df["total_bill"] > 130)])
