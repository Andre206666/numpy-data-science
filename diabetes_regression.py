import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

diabetes = load_diabetes(as_frame=True)
df = diabetes.frame

print("Shape of the dataset:")
print(df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values check:")
print(df.isna().sum())

X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred_lr = model.predict(X_test)

comparison_df = pd.DataFrame({
    "Actual": y_test.iloc[:5].values,
    "Predicted": y_pred_lr[:5],
    "Difference": y_test.iloc[:5].values - y_pred_lr[:5]
})
print(comparison_df)


lr_mae = mean_absolute_error(y_test, y_pred_lr)
lr_mse = mean_squared_error(y_test, y_pred_lr)
lr_r2 = r2_score(y_test, y_pred_lr)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

rf_mae = mean_absolute_error(y_test, y_pred_rf)
rf_mse = mean_squared_error(y_test, y_pred_rf)
rf_r2 = r2_score(y_test, y_pred_rf)

results = pd.DataFrame({
    "Metric": ["MAE", "MSE", "R² Score"],
    "Linear Regression": [f"{lr_mae:.4f}", f"{lr_mse:.4f}", f"{lr_r2:.4f}"],
    "Random Forest": [f"{rf_mae:.4f}", f"{rf_mse:.4f}", f"{rf_r2:.4f}"]
})

print(results)