"""
Step 6
     Take engineered features (lags, rolling averages, targets)
      and push them into Hopsworks as a NEW Feature Group, separate
      from the raw one. This keeps raw data untouched/reusable, while
      giving us a clean "ready to train" table.
"""

import hopsworks
import pandas as pd

print("Loading engineered features from local checkpoint...")
df = pd.read_csv("data/engineered_features.csv")

# Hopsworks needs a proper datetime column for event_time, same as Day 6
df["date"] = pd.to_datetime(df["date"])

print("Shape being pushed to Hopsworks:", df.shape)
print(df.dtypes)

print("\nLogging in to Hopsworks...")
project = hopsworks.login()
fs = project.get_feature_store()

# Create (or get, if it already exists) the new Feature Group
engineered_fg = fs.get_or_create_feature_group(
    name="aqi_engineered_features",
    version=1,
    description="Engineered AQI features: lags, rolling averages, "
                 "calendar features, and 1/2/3-day PM2.5 targets, per city.",
    primary_key=["city", "date"],
    event_time="date",
    time_travel_format="HUDI",  # same as Day 6, avoids extra delta library issue
)

print("\nInserting data into Hopsworks (this may take a minute)...")
engineered_fg.insert(df)

print("\nDone! Engineered features are now stored in Hopsworks.")
print(f"View it here: https://eu-west.cloud.hopsworks.ai:443/p/{project.id}")