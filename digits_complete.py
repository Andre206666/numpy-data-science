from sklearn.datasets import load_digits
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report


digits = load_digits()
X = pd.DataFrame(digits.data)
y = digits.target

print(X.shape)
print(y[:10])
print(pd.Series(y).value_counts())

X_train, X_test, y_train, y_test = train_test_split(X, y)

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
print(f"RF {accuracy_score(y_test, rf.predict(X_test))}")

lr = LogisticRegression()
lr.fit(X_train, y_train)
print(f"LR {accuracy_score(y_test, lr.predict(X_test))}")

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
print(f"DT {accuracy_score(y_test, dt.predict(X_test))}")

param_grid = {"n_estimators": [20, 40, 80], "max_depth": [2, 4, 8, None]}
grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
predictions = best_model.predict(X_test)

print(f"\nAcuuracy {accuracy_score(y_test, predictions)}")
print(f"\nClassificatin report")
print(classification_report(y_test, predictions))

print(best_model.predict(X_test.iloc[[0]]))
print(best_model.predict(X_test.iloc[[1]]))
print(best_model.predict(X_test.iloc[[2]]))

X["predicted"] = best_model.predict(X)
X.to_csv("digits.complete.csv", index=False)
print("Saved!")


