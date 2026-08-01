import hopsworks
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

project = hopsworks.login()
fs = project.get_feature_store()
fv = fs.get_feature_view(name="aqi_fv_1d", version=1)

X_train, X_test, y_train, y_test = fv.train_test_split(
    train_start="2024-01-01",
    train_end="2026-05-31",
    test_start="2026-06-01",
    test_end="2026-07-24"
)

y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

# Add a momentum feature: avg daily change over last 3 days
X_train = X_train.copy()
X_test = X_test.copy()
X_train["aqi_change_rate"] = (X_train["pm25_lag_1"] - X_train["pm25_lag_3"]) / 3
X_test["aqi_change_rate"] = (X_test["pm25_lag_1"] - X_test["pm25_lag_3"]) / 3

# \One-hot encode city, drop date 
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


lr = LinearRegression()
lr.fit(X_train_model, y_train)
lr_preds = lr.predict(X_test_model)
evaluate("Linear Regression (1d)", y_test, lr_preds)

# Hyperparameter search for Random Forest 
tscv = TimeSeriesSplit(n_splits=3)
param_grid = {
    "n_estimators": [200, 400],
    "max_depth": [8, 12, None],
    "min_samples_leaf": [1, 3, 5]
}
rf_search = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid,
    cv=tscv,
    scoring="r2",
    n_jobs=-1
)
rf_search.fit(X_train_model, y_train)
print("Best RF params:", rf_search.best_params_)

rf_best = rf_search.best_estimator_
rf_preds = rf_best.predict(X_test_model)
evaluate("Random Forest tuned (1d)", y_test, rf_preds)

# Per-city error breakdown 
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

# Random Forest trained on log-transformed target
y_train_log = np.log1p(y_train)

rf_log = RandomForestRegressor(n_estimators=400, max_depth=8, min_samples_leaf=3, random_state=42)
rf_log.fit(X_train_model, y_train_log)

rf_log_preds = np.expm1(rf_log.predict(X_test_model))
evaluate("Random Forest (log-target)", y_test, rf_log_preds)


#  Save model locally first (Hopsworks registry needs a file)


os.makedirs("models", exist_ok=True)
joblib.dump(lr, "models/model_1d.pkl")

# Register model in Hopsworks Model Registry 
mr = project.get_model_registry()

model_1d = mr.python.create_model(
    name="aqi_model_1d",
    metrics={"mae": 8.48, "rmse": 12.94, "r2": 0.650},
    description="Linear Regression model predicting next-day PM2.5 for 5 Pakistani cities"
)
model_1d.save("models/model_1d.pkl")

print("Model registered successfully in Hopsworks Model Registry.")