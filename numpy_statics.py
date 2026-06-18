import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

np.random.seed(42)

group_a = np.random.normal(85, 5, 50)
group_b = np.random.normal(55, 10, 50)

print("Group A mean:", np.mean(group_a))
print("Group A std:", np.std(group_a))
print("Group B mean:", np.mean(group_b))
print("Group B std:", np.std(group_b))

df_groups = pd.DataFrame({
    "score": np.concatenate([group_a, group_b]),
    "group": ["A"] * 50 + ["B"] * 50
})

sns.histplot(data=df_groups, x="score", hue="group", bins=20)
plt.title("Group A vs Group B")
plt.show()