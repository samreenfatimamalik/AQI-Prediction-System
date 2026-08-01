"""
 Step 7
Create 3 Feature Views on top of our engineered Feature Group.
      A Feature View = a saved "recipe" telling Hopsworks:
        - which columns are INPUTS (features)
        - which single column is the ANSWER (label)
      We make 3 separate views because we're training 3 separate
      models later (1-day, 2-day, 3-day ahead predictions).

"""

import hopsworks

print("Logging in to Hopsworks...")
project = hopsworks.login()
fs = project.get_feature_store()

# Get the engineered Feature Group we just created (Step 6)
engineered_fg = fs.get_feature_group(name="aqi_engineered_features", version=1)

# Columns that are INPUT features (used for all 3 views, same inputs each time)
input_features = [
    "city", "date",
    "pm2_5", "pm10", "temperature", "humidity", "wind_speed", "pressure",
    "day_of_week", "month",
    "pm25_lag_1", "pm25_lag_3", "pm25_lag_7",
    "pm25_rolling_3", "pm25_rolling_7",
]

targets = {
    "target_1d": "aqi_fv_1d",
    "target_2d": "aqi_fv_2d",
    "target_3d": "aqi_fv_3d",
}

for target_col, view_name in targets.items():
    print(f"\nCreating Feature View: {view_name} (label = {target_col})")

    # Select input features + ONLY this one target column
    query = engineered_fg.select(input_features + [target_col])

    feature_view = fs.get_or_create_feature_view(
        name=view_name,
        version=1,
        query=query,
        labels=[target_col],  # tells Hopsworks: this column is the answer key
    )
    print(f"Created: {view_name}, pointing to label '{target_col}'")

print("\nAll 3 Feature Views created successfully.")
