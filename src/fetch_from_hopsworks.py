import hopsworks
import pandas as pd

print("SCRIPT STARTED")

# 1. Log in to your Hopsworks project 
print("Logging in to Hopsworks...")
project = hopsworks.login()
print("Login successful!")

fs = project.get_feature_store()

aqi_fg = fs.get_feature_group(name="aqi_daily_features", version=1)

aqi_fg = fs.get_feature_group(name="aqi_daily_features", version=1)

# OLD (Flight-dependent, timing out):
# df = aqi_fg.read()

# NEW (bypasses Flight, uses reliable Hive path):
try:
    df = aqi_fg.read()
except Exception as e:
    print(f"Flight read failed ({e}), falling back to Hive...")
    df = aqi_fg.select_all().read(read_options={"use_hive": True})

print("Shape of data pulled from Hopsworks:", df.shape)
print(df.head())
print(df["city"].unique())


print("Shape of data pulled from Hopsworks:", df.shape)

# 5. CRITICAL: sort by city first, then by date within each city.
#    Why? Hopsworks does NOT guarantee row order. If we don't sort,
#    "yesterday's value" (lag feature) could accidentally come from
#    a totally different city or a random date. Sorting fixes this.
df = df.sort_values(by=["city", "date"]).reset_index(drop=True)

print("\nAfter sorting (check Lahore rows are now in date order):")
print(df[df["city"] == "Lahore"].head(10)[["date", "city", "pm2_5"]])

# Save this cleaned, sorted version locally too, just so we have
# a checkpoint to look at (not the final storage - Hopsworks stays
# the source of truth, this is just for our own sanity-checking)
df.to_csv("data/raw_from_hopsworks_sorted.csv", index=False)
print("\nSaved sorted checkpoint to data/raw_from_hopsworks_sorted.csv")



# STEP 3: Feature Engineering (per city, to avoid data leakage)


# 6. Calendar features - simple, free signals
#    day_of_week: 0=Monday ... 6=Sunday
#    month: 1-12 (helps capture seasonal patterns like winter smog)
df["day_of_week"] = df["date"].dt.dayofweek.astype("int64")
df["month"] = df["date"].dt.month.astype("int64")

# 7. Lag features - "what was PM2.5 N days ago, for THIS city"
#    groupby("city") ensures Lahore only looks at Lahore's own past,
#    never accidentally grabs a value from Karachi/Islamabad/etc.
df["pm25_lag_1"] = df.groupby("city")["pm2_5"].shift(1)
df["pm25_lag_3"] = df.groupby("city")["pm2_5"].shift(3)
df["pm25_lag_7"] = df.groupby("city")["pm2_5"].shift(7)

# 8. Rolling averages - smoothed recent trend, per city
#    .shift(1) inside rolling ensures we only use PAST days,
#    never today's own value (that would be cheating/leakage)
df["pm25_rolling_3"] = (
    df.groupby("city")["pm2_5"]
    .shift(1)
    .rolling(window=3)
    .mean()
    .reset_index(level=0, drop=True)
)
df["pm25_rolling_7"] = (
    df.groupby("city")["pm2_5"]
    .shift(1)
    .rolling(window=7)
    .mean()
    .reset_index(level=0, drop=True)
)

print("\nSample engineered features for Lahore:")
print(df[df["city"] == "Lahore"][
    ["date", "pm2_5", "pm25_lag_1", "pm25_lag_7", "pm25_rolling_3", "pm25_rolling_7"]
].head(10))

print("\nMissing values per column (expected at the START of each city's data):")
print(df.isna().sum())




# STEP 4: Create prediction targets (what the model should predict)


# .shift(-1) looks FORWARD instead of backward - "tomorrow's value"
# groupby("city") again ensures Lahore's "tomorrow" never comes from
# another city's row.
df["target_1d"] = df.groupby("city")["pm2_5"].shift(-1)
df["target_2d"] = df.groupby("city")["pm2_5"].shift(-2)
df["target_3d"] = df.groupby("city")["pm2_5"].shift(-3)

print("\nSample targets for Lahore (check target_1d = next row's pm2_5):")
print(df[df["city"] == "Lahore"][
    ["date", "pm2_5", "target_1d", "target_2d", "target_3d"]
].tail(10))

# The LAST few rows of each city will have empty targets - that's expected
# (there's no "tomorrow" for the very last day we have data for).
# We must drop those rows before training, since a model can't learn
# from a row where the answer key is blank.
before = len(df)
df_clean = df.dropna(subset=["target_1d", "target_2d", "target_3d",
                              "pm25_lag_7", "pm25_rolling_7"])
after = len(df_clean)
print(f"\nRows before dropping incomplete ones: {before}")
print(f"Rows after dropping incomplete ones:  {after}")
print(f"Dropped {before - after} rows (expected: start+end gaps per city)")

df_clean.to_csv("data/engineered_features.csv", index=False)
print("\nSaved final engineered dataset to data/engineered_features.csv")


# STEP 5: Push engineered features to Hopsworks Feature Store


print("\nPushing engineered features to Hopsworks...")

engineered_fg = fs.get_or_create_feature_group(
    name="aqi_engineered_features",
    version=1,
    primary_key=["city", "date"],
    event_time="date",
    time_travel_format="HUDI",
    description="Engineered AQI features (lags, rolling averages, targets) for 5 Pakistani cities",
)

engineered_fg.insert(df_clean, write_options={"wait_for_job": True})

print("Successfully pushed", len(df_clean), "rows to aqi_engineered_features in Hopsworks.")