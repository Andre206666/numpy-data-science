import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

np.random.seed(42)
n = 200

data = {
    "size": np.random.randint(50, 100, n),
    "rooms": np.random.randint(1, 6, n),
    "age": np.random.randint(1, 50, n),
    "distance": np.random.randint(1, 30, n),
    "price": None

}

df = pd.DataFrame(data)
df["price"] = (df["size"] * 1000 +
               df["rooms"] * 5000 -
               df["age"] * 500 -
               df["distance"] * 2000 +
               np.random.normal(0, 20000,n))

print(df.head())
print(df.describe())

sns.heatmap(df[["size", "rooms", "distance", "price"]].corr(), annot=True, cmap="coolwarm")
plt.title("All columns")
plt.show()

sns.scatterplot(data=df, x="size", y="price", hue="rooms")
plt.title("Size vs Price")
plt.show()

X = df[["size", "rooms", "age", "distance"]].values
y = df["price"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)
print(f"R²: {model.score(X_test, y_test):.2f}")

prediction = model.predict([[150, 3, 10, 5]])
print(f"Predicted price: ${prediction[0]:,.0f}")


plt.scatter(y_test, model.predict(X_test))
plt.xlabel("Real price")
plt.ylabel("Predicted price")
plt.title("Real vs Predicted")
plt.show()