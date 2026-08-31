from sklearn.datasets import load_wine
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = pd.Series(wine.target)

print(X.head())
print(y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

RF = RandomForestClassifier()
RF.fit(X_test, y_test)
print(f"RF {accuracy_score(y_test, RF.predict(X_test))}")
print(classification_report(y_test, RF.predict(X_test)))


