import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

X_raw, y_raw = make_regression(n_samples=500, n_features=15, n_informative=5, noise=10.0, random_state=42)

feature_names = [f"feature_{i}" for i in range(15)]
df = pd.DataFrame(X_raw, columns=feature_names)
df["target"] = y_raw

print(df.shape)
print(df.head())

X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LinearRegression().fit(X_train, y_train)
ridge = Ridge().fit(X_train, y_train)
lasso = Lasso().fit(X_train, y_train)
rf = RandomForestRegressor(random_state=42).fit(X_train, y_train)

coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Linear": lr.coef_,
    "Ridge (L2)": ridge.coef_,
    "Lasso (L1)": lasso.coef_
})
print(f"==\nFeature coefficients ==")
print(coef_df.round(2).to_string(index=False))

models = {
    "Linear Regression": lr,
    "Ridge Regression": ridge,
    "Lasso Regression": lasso,
    "Random Forest": rf
}
results = []
for name, model in models.items():
    preds = model.predict(X_test)
    results.append({
        "Model": name,
        "MAE": mean_absolute_error(y_test, preds),
        "MSE": mean_squared_error(y_test, preds),
        "R2": r2_score(y_test, preds)
    })
print(f"Model performance comparison")
print(pd.DataFrame(results).to_string(index=False))