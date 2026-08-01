import hopsworks
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

project = hopsworks.login()
fs = project.get_feature_store()
fv = fs.get_feature_view(name="aqi_fv_3d", version=1)

X_train, X_test, y_train, y_test = fv.train_test_split(
    train_start="2024-01-01",
    train_end="2026-05-31",
    test_start="2026-06-01",
    test_end="2026-07-24"
)

y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

X_train = X_train.copy()
X_test = X_test.copy()

# Momentum feature: avg daily change over last 3 days
X_train["aqi_change_rate"] = (X_train["pm25_lag_1"] - X_train["pm25_lag_3"]) / 3
X_test["aqi_change_rate"] = (X_test["pm25_lag_1"] - X_test["pm25_lag_3"]) / 3

# Volatility: how unstable has pollution been lately (spikes follow instability)
X_train["volatility_7"] = (X_train["pm25_lag_1"] - X_train["pm25_rolling_7"]).abs()
X_test["volatility_7"] = (X_test["pm25_lag_1"] - X_test["pm25_rolling_7"]).abs()

# Stagnation index: low wind + high humidity traps pollution near the ground
X_train["wind_stagnation"] = X_train["humidity"] / (X_train["wind_speed"] + 1)
X_test["wind_stagnation"] = X_test["humidity"] / (X_test["wind_speed"] + 1)

# Pressure-humidity interaction: high pressure + high humidity = trapped smog conditions
X_train["pressure_humidity"] = X_train["pressure"] * X_train["humidity"] / 1000
X_test["pressure_humidity"] = X_test["pressure"] * X_test["humidity"] / 1000


# One-hot encode city, drop date
X_train_model = pd.get_dummies(X_train.drop(columns=["date"]), columns=["city"])
X_test_model = pd.get_dummies(X_test.drop(columns=["date"]), columns=["city"])
X_train_model, X_test_model = X_train_model.align(X_test_model, join="left", axis=1, fill_value=0)

print("Feature columns:", list(X_train_model.columns))
print()

def evaluate(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"{name}: MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.3f}")
    return mae, rmse, r2


lr = LinearRegression()
lr.fit(X_train_model, y_train)
lr_preds = lr.predict(X_test_model)
lr_mae, lr_rmse, lr_r2 = evaluate("Linear Regression (3d)", y_test, lr_preds)

# Hyperparameter search for Random Forest
tscv = TimeSeriesSplit(n_splits=3)
rf_param_grid = {
    "n_estimators": [200, 400],
    "max_depth": [8, 12, None],
    "min_samples_leaf": [1, 3, 5]
}
rf_search = GridSearchCV(
    RandomForestRegressor(random_state=42),
    rf_param_grid,
    cv=tscv,
    scoring="r2",
    n_jobs=-1
)
rf_search.fit(X_train_model, y_train)
print("Best RF params:", rf_search.best_params_)

rf_best = rf_search.best_estimator_
rf_preds = rf_best.predict(X_test_model)
rf_mae, rf_rmse, rf_r2 = evaluate("Random Forest tuned (3d)", y_test, rf_preds)

# HistGradientBoostingRegressor — often stronger than RF on tabular data with interactions
hgb_param_grid = {
    "max_depth": [4, 6, None],
    "max_iter": [150, 300],
    "learning_rate": [0.05, 0.1]
}
hgb_search = GridSearchCV(
    HistGradientBoostingRegressor(random_state=42),
    hgb_param_grid,
    cv=tscv,
    scoring="r2",
    n_jobs=-1
)
hgb_search.fit(X_train_model, y_train)
print("Best HGB params:", hgb_search.best_params_)

hgb_best = hgb_search.best_estimator_
hgb_preds = hgb_best.predict(X_test_model)
hgb_mae, hgb_rmse, hgb_r2 = evaluate("HistGradientBoosting tuned (3d)", y_test, hgb_preds)

# Per-city error breakdown (using the tuned RF as reference)
results_df = X_test.copy()
results_df["actual"] = y_test
results_df["predicted"] = rf_preds
results_df["error"] = results_df["actual"] - results_df["predicted"]

print("\n--- Per-city MAE ---")
print(results_df.groupby("city")["error"].apply(lambda e: e.abs().mean()))

print("\n--- Target distribution ---")
print(pd.Series(y_train).describe())

print("\n--- Biggest misses ---")
print(results_df.reindex(results_df["error"].abs().sort_values(ascending=False).index)[["city","date","actual","predicted","error"]].head(10))

# Random Forest trained on log-transformed target (kept for comparison/diagnostics only)
y_train_log = np.log1p(y_train)
rf_log = RandomForestRegressor(n_estimators=400, max_depth=8, min_samples_leaf=3, random_state=42)
rf_log.fit(X_train_model, y_train_log)
rf_log_preds = np.expm1(rf_log.predict(X_test_model))
evaluate("Random Forest (log-target)", y_test, rf_log_preds)



# CHOOSE BEST MODEL — auto-pick the highest R2 among all candidates
# (fixed: no longer hardcoded to Linear Regression)

candidates = {
    "Linear Regression": (lr, lr_mae, lr_rmse, lr_r2),
    "Random Forest (tuned)": (rf_best, rf_mae, rf_rmse, rf_r2),
    "HistGradientBoosting (tuned)": (hgb_best, hgb_mae, hgb_rmse, hgb_r2),
}
best_model_desc = max(candidates, key=lambda k: candidates[k][3])
best_model, best_mae, best_rmse, best_r2 = candidates[best_model_desc]
best_metrics = {"mae": float(best_mae), "rmse": float(best_rmse), "r2": float(best_r2)}
print(f"\n>>> WINNER: {best_model_desc} with R2={best_r2:.3f} <<<\n")

# Save model locally first (Hopsworks registry needs a file)
os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/model_3d.pkl")

# Register model in Hopsworks Model Registry
mr = project.get_model_registry()

model_3d = mr.python.create_model(
    name="aqi_model_3d",
    metrics=best_metrics,
    description=f"{best_model_desc} model predicting 3-day-ahead PM2.5 for 5 Pakistani cities"
)
model_3d.save("models/model_3d.pkl")

print("Model registered successfully in Hopsworks Model Registry.")