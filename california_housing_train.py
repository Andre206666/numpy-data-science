import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

housing = fetch_california_housing(as_frame=True)
df = housing.frame

print(df.head())
print(df.shape)
print(df.isna().sum())

X = df.drop(columns=["MedHouseVal"])
y = df["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

y_pred_lr = lr_model.predict(X_test)

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
print(results.to_string(index=False))