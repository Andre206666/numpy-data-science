from sklearn.datasets import load_wine
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

wine = load_wine()
x = pd.DataFrame(wine.data, columns=wine.feature_names)
y = wine.target

models = {
    "LR": LogisticRegression(max_iter=1000),
    "DT": DecisionTreeClassifier(),
    "RF": RandomForestClassifier()
}

for name, model in models.items():
    scores = cross_val_score(model, x, y, cv=5)
    print(f"\n{name}: {scores.mean():.2f}")
