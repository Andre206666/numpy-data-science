from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
df = pd.read_csv("https://raw.githubusercontent.com/dsrscientist/DSData/master/winequality-red.csv")
df["good_wine"] = df["quality"].apply(lambda x: 1 if x >= 7 else 0)

X = df.drop(columns=["good_wine", "quality"])
y = df["good_wine"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


lr = LogisticRegression(max_iter=1000)
dt = DecisionTreeClassifier(max_depth=5)
rf = RandomForestClassifier(n_estimators=100)

voting_model = VotingClassifier(
    estimators=[("lr", lr), ("dt", dt), ("rf", rf)],
    voting="soft"
)

voting_model.fit(X_train, y_train)
print(f"Voting accuracy: {voting_model.score, voting_model.predict(X_test):.2f}")
