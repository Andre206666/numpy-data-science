from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
from sklearn.metrics import roc_auc_score, RocCurveDisplay

import pandas as pd
df = pd.read_csv("https://raw.githubusercontent.com/dsrscientist/DSData/master/winequality-red.csv")
df["good_wine"] = df["quality"].apply(lambda x: 1 if x >= 7 else 0)

X = df.drop(columns=["good_wine", "quality"])
y = df["good_wine"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svm_rbf = SVC(kernel="rbf")
svm_rbf.fit(X_train, y_train)

predictions = svm_rbf.predict(X_test)
print(f"SVM Accuracy: {accuracy_score(y_test, predictions):.2f}")


svm2_rbf = SVC(kernel="linear")
svm2_rbf.fit(X_train, y_train)

predictions = svm2_rbf.predict(X_test)
print(f"SVM2 Accuracy: {accuracy_score(y_test, predictions)}")


