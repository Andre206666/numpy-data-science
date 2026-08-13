import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix
)

import joblib


df = pd.read_csv(
    "https://raw.githubusercontent.com/nelson-wu/employee-attrition-ml/master/WA_Fn-UseC_-HR-Employee-Attrition.csv"
)

print(df.head())
print(df.shape)

print("\n--- INFO ---")
print(df.info())

print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

print("\n--- DUPLICATES ---")
print(df.duplicated().sum())

print("\n--- ATTRITION ---")
print(df["Attrition"].value_counts())

print("\n--- ATTRITION % ---")
print(df["Attrition"].value_counts(normalize=True) * 100)

df["Attrition"].value_counts().plot(kind="bar")
plt.title("Employee Attrition")
plt.xlabel("Attrition")
plt.ylabel("Employees")
plt.show()


df = df.drop_duplicates()

df = df.drop(
    ["EmployeeNumber", "EmployeeCount", "Over18", "StandardHours"],
    axis=1
)


df["YearlyIncome"] = df["MonthlyIncome"] * 12

df["YearsWithoutPromotion"] = (
    df["YearsAtCompany"] - df["YearsSinceLastPromotion"]
)


df = pd.get_dummies(df, drop_first=True)


X = df.drop("Attrition_Yes", axis=1)

y = df["Attrition_Yes"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ),

    "Decision Tree": DecisionTreeClassifier(
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        class_weight="balanced",
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        random_state=42
    ),

    "SVM": SVC(
        class_weight="balanced"
    )
}


results = []

for name, model in models.items():

    print("\nTraining:", name)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall
    })


results_df = pd.DataFrame(results)

print("\n--- MODEL RESULTS ---")

print(
    results_df.sort_values(
        by="Recall",
        ascending=False
    )
)


parameters = {
    "n_estimators": [100, 200],
    "max_depth": [5, 10, None]
}


grid = GridSearchCV(
    RandomForestClassifier(
        class_weight="balanced",
        random_state=42
    ),
    parameters,
    cv=5,
    scoring="recall"
)

grid.fit(X_train, y_train)


print("\n--- GRID SEARCH ---")

print("Best parameters:")
print(grid.best_params_)

print("Best recall:")
print(grid.best_score_)


best_model = grid.best_estimator_

predictions = best_model.predict(X_test)


print("\n--- BEST MODEL RESULTS ---")

print(
    "Accuracy:",
    accuracy_score(y_test, predictions)
)

print(
    "Precision:",
    precision_score(y_test, predictions)
)

print(
    "Recall:",
    recall_score(y_test, predictions)
)


print("\n--- CLASSIFICATION REPORT ---")

print(
    classification_report(
        y_test,
        predictions
    )
)


print("\n--- CONFUSION MATRIX ---")

cm = confusion_matrix(
    y_test,
    predictions
)

print(cm)


importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": best_model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n--- TOP FEATURES ---")

print(importance.head(15))


importance.head(15).plot(
    x="Feature",
    y="Importance",
    kind="barh",
    figsize=(10, 7),
    legend=False
)

plt.title("Top Factors Associated With Employee Attrition")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.gca().invert_yaxis()
plt.show()


joblib.dump(
    best_model,
    "employee_attrition_model.pkl"
)

print("\nModel saved successfully!")