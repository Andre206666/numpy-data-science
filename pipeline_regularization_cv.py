import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor


diabetes = load_diabetes(as_frame=True)

X = diabetes.frame.drop(columns=["target"])
y = diabetes.frame["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"X_train scaled shape: {X_train_scaled.shape}")
print(f"X_test scaled shape: {X_test_scaled.shape}")
print(f"Trains means: {X_train_scaled.mean(axis=0).round(2)}")
print(f"Tests std: {X_test_scaled.std(axis=0).round(2)}")

alphas = np.logspace(-3, 2, 50)

lr = LinearRegression().fit(X_train_scaled, y_train)

ridge_cv = RidgeCV(alphas=alphas, cv=5).fit(X_train_scaled, y_train)

lasso_cv = LassoCV(alphas=alphas, cv=5, random_state=42).fit(X_train_scaled, y_train)

rf = RandomForestRegressor(random_state=42).fit(X_train_scaled, y_train)

print("\n=== OPTIMAL HYPERPARAMETERS FOUND ===")
print(f"Optimal Alpha for Ridge: {ridge_cv.alpha_:.4f}")
print(f"Optimal Alpha for Lasso: {lasso_cv.alpha_:.4f}")

coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Linea": lr.coef_,
    "Ridge": ridge_cv.coef_,
    "Lasso": lasso_cv.coef_
})

print(f"===\nFeature coefficients ===")
print(coef_df.round(2).to_string(index=False))

models = {
    "Linear Regression": lr,
    "Ridge Regression": ridge_cv,
    "Lasso Regression": lasso_cv,
    "Random Forest": rf
}

results = []
for name, model in models.items():
    preds = model.predict(X_test_scaled)
    results.append({
        "Model": name,
        "MAE": mean_absolute_error(y_test, preds),
        "MSE": mean_squared_error(y_test, preds),
        "R2": r2_score(y_test, preds)
    })
print(f"Model performance comparison")
print(pd.DataFrame(results).round(2).to_string(index=False))