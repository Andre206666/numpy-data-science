from sklearn.linear_model import LinearRegression
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

hours = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
scores = np.array([50, 55, 60, 65, 70, 75, 80, 85, 90, 95])

model = LinearRegression()

model.fit(hours, scores)

prediction = model.predict([[7.5]])
print(f"Predictes score for 7.5 hours {prediction[0]:.1f}")

print(f"Model accuracy (R²): {model.score(hours, scores):.2f}")


plt.figure(figsize=(8, 5))
plt.scatter(hours, scores, color="blue", label="Real data")
plt.plot(hours, model.predict(hours), color="red", label="Prediction line")
plt.title("Hours Studied vs Score")
plt.xlabel("Hours")
plt.ylabel("Score")
plt.legend()
plt.show()

np.random.seed(42)

hours_real = np.random.randint(1, 10, 50).reshape(-1, 1)
scores_real = hours_real.flatten() * 5 + 45 + np.random.normal(0, 10, 50)

model2 = LinearRegression()
model2.fit(hours_real, scores_real)

print(f"R²: {model2.score(hours_real, scores_real):.2f}")

plt.figure(figsize=(8, 5))
plt.scatter(hours_real, scores_real, color="blue", alpha=0.5)
plt.plot(hours_real, model2.predict(hours_real), color="red")
plt.title("Real World Data - Hours vs Score")
plt.show()

np.random.seed(42)

experience = np.random.randint(1, 30, 100).reshape(-1, 1)
salary = experience.flatten() * 2000 + 3000 + np.random.normal(0, 5000, 100)

model3 = LinearRegression()
model3.fit(experience, salary)
print(f"R²: {model3.score(experience, salary):.2f}")

prediction = model3.predict([[15]])
print(f"Predicted salary for 15 years: ${prediction[0]:,.0f}")

plt.figure(figsize=(8, 5))
plt.scatter(experience, salary, color="pink", alpha=0.5)
plt.plot(experience, model3.predict(experience), color="green")
plt.title("Experience vs Salary")
plt.show()

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

np.random.seed(42)

hours = np.random.randint(1, 10, 100).reshape(-1, 1)
passed = (hours.flatten() * 10 + np.random.normal(0, 15, 100) > 50).astype(int)

X_train, X_test, y_train, y_test = train_test_split(hours, passed, test_size=0.2)

model = LogisticRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.2f}")

print(f"5 hours → {'Pass' if model.predict([[5]])[0] == 1 else 'Fail'}")
print(f"8 hours → {'Pass' if model.predict([[8]])[0] == 1 else 'Fail'}")

np.random.seed(42)

age = np.random.randint(20, 80, 100)
bmi = np.random.randint(18, 40, 100)

diabetes = ((age > 45) & (bmi > 28)).astype(int)
X = np.column_stack([age, bmi])
X_train, X_test, y_train, y_test = train_test_split(X, diabetes, test_size=0.2)

model_diabetes = LogisticRegression()
model_diabetes.fit(X_train, y_train)

print(f"Accuracy: {model_diabetes.score(X_test, y_test):.2f}")

print(model_diabetes.predict([[45, 30]]))
print(model_diabetes.predict([[30, 20]]))


