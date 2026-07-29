import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report



df = pd.read_csv("https://raw.githubusercontent.com/dsrscientist/DSData/master/winequality-red.csv")
print(df.head())
print(df["quality"].value_counts())

df["good_wine"] = df["quality"].apply(lambda x: 1 if x >= 7 else 0)
print(df[["good_wine", "quality"]].head())

plt.figure(figsize=(12,10))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.show()

X = df.drop(columns=["good_wine", "quality"], errors="ignore")
y = df["good_wine"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LogisticRegression()
lr.fit(X_train, y_train)
print(f"LR {accuracy_score(y_test, lr.predict(X_test))}")

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
print(f"RF {accuracy_score(y_test, rf.predict(X_test))}")

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
print(f"DT {accuracy_score(y_test, dt.predict(X_test))}")


base_model = RandomForestClassifier(random_state=42)

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5],
    'criterion': ['gini', 'entropy']
}

grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print(f"Best CV Score: {grid_search.best_score_:.4f}")

print("Class counts:\n", y_train.value_counts())
print("\nClass percentages:\n", y_train.value_counts(normalize=True) * 100)


best_params = grid_search.best_params_

final_model = RandomForestClassifier(
    **best_params,
    class_weight='balanced',
    random_state=42
)

final_model.fit(X_train, y_train)

print("Final model successfully trained with balanced class weights!")

importance = pd.DataFrame({
    "feature": X.columns,
    "importance": final_model.feature_importances_
}).sort_values("importance", ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(data=importance, x="importance", y="feature")
plt.title("Feature Importance - Wine Quality")
plt.show()

predictions = final_model.predict(X_test)
print(classification_report(y_test, predictions))

X_test_copy = X_test.copy()
X_test_copy["actual"] = y_test
X_test_copy["predicted"] = predictions
X_test_copy.to_csv("wine_quality_results.csv", index=False)
print("Saved!")

importance = pd.DataFrame({
    "feature": X.columns,
    "importance": final_model.feature_importances_
}).sort_values("importance", ascending=False)

print(importance)