import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv")

df["bmi_category"] = df["BMI"].apply(lambda x: "Low" if x < 18.5 else ("Normal" if x <= 25 else "High"))
print(df[["BMI", "bmi_category"]].head())

df['age_group'] = df['Age'].apply(lambda x: 'Young' if x < 30 else ('Middle' if x <= 50 else 'Senior'))
print(df[["Age", "age_group"]].head())

X = df[["BMI", "bmi_category", "Age", "age_group"]]
X = pd.get_dummies(X, columns=["bmi_category", "age_group"], drop_first=True)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 10]
}

model = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring="accuracy")
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_}")

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Final Test Accuracy: {accuracy:.4f}\n")

print("Classification Report:")
print(classification_report(y_test, y_pred))

rf_balanced = RandomForestClassifier(n_estimators=100, max_depth=3, class_weight="balanced")
rf_balanced.fit(X_train, y_train)

predictions_balanced = rf_balanced.predict(X_test)
print(f"Balanced accuracy: {accuracy_score(y_test, predictions_balanced):.2f}")
print(classification_report(y_test, predictions_balanced))
