import hopsworks
import pandas as pd
import os
import time
import sys

print("SCRIPT STARTED")


def read_fg_with_retry(fg, max_retries=3, wait_seconds=30):
    """Try reading a Feature Group with retries, then fall back to Hive,
    then give up cleanly (no ugly crash) if Hopsworks itself is down."""
    for attempt in range(1, max_retries + 1):
        try:
            return fg.read()
        except Exception as e:
            print(f"[Flight attempt {attempt}/{max_retries}] failed: {e}")
            if attempt < max_retries:
                print(f"Waiting {wait_seconds}s before retry...")
                time.sleep(wait_seconds)

    print("All Flight retries failed. Falling back to Hive read path...")
    try:
        return fg.select_all().read(read_options={"use_hive": True})
    except Exception as e:
        print(f"Hive fallback also failed: {e}")
        print("Hopsworks Feature Query Service appears to be down right now. "
              "Skipping this run — next scheduled run will try again.")
        sys.exit(1)


# 1. Log in to your Hopsworks project
print("Logging in to Hopsworks...")
project = hopsworks.login()
print("Login successful!")

fs = project.get_feature_store()

aqi_fg = fs.get_feature_group(name="aqi_daily_features", version=1)

df = read_fg_with_retry(aqi_fg)

print("Shape of data pulled from Hopsworks:", df.shape)
print(df.head())
print(df["city"].unique())

# 5. CRITICAL: sort by city first, then by date within each city.
df = df.sort_values(by=["city", "date"]).reset_index(drop=True)

print("\nAfter sorting (check Lahore rows are now in date order):")
print(df[df["city"] == "Lahore"].head(10)[["date", "city", "pm2_5"]])

os.makedirs("data", exist_ok=True)
df.to_csv("data/raw_from_hopsworks_sorted.csv", index=False)
print("\nSaved sorted checkpoint to data/raw_from_hopsworks_sorted.csv")


# STEP 3: Feature Engineering (per city, to avoid data leakage)

df["day_of_week"] = df["date"].dt.dayofweek.astype("int64")
df["month"] = df["date"].dt.month.astype("int64")

df["pm25_lag_1"] = df.groupby("city")["pm2_5"].shift(1)
df["pm25_lag_3"] = df.groupby("city")["pm2_5"].shift(3)
df["pm25_lag_7"] = df.groupby("city")["pm2_5"].shift(7)

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


# STEP 4: Create prediction targets

df["target_1d"] = df.groupby("city")["pm2_5"].shift(-1)
df["target_2d"] = df.groupby("city")["pm2_5"].shift(-2)
df["target_3d"] = df.groupby("city")["pm2_5"].shift(-3)

print("\nSample targets for Lahore (check target_1d = next row's pm2_5):")
print(df[df["city"] == "Lahore"][
    ["date", "pm2_5", "target_1d", "target_2d", "target_3d"]
].tail(10))

before = len(df)
df_clean = df.dropna(subset=["pm25_lag_7", "pm25_rolling_7"])
after = len(df_clean)
print(f"\nRows before dropping incomplete ones: {before}")
print(f"Rows after dropping incomplete ones:  {after}")
print(f"Dropped {before - after} rows (expected: start-of-city gaps only)")

df_clean.to_csv("data/engineered_features.csv", index=False)
print("\nSaved final engineered dataset to data/engineered_features.csv")


# STEP 5: Push engineered features to Hopsworks Feature Store

# STEP 5: Push engineered features to Hopsworks Feature Store

print("\nPushing engineered features to Hopsworks...")

engineered_fg = fs.get_or_create_feature_group(
    name="aqi_engineered_features",
    version=1,
    primary_key=["city", "date"],
    event_time="date",
    time_travel_format="HUDI",
    description="Engineered AQI features (lags, rolling averages, targets) for 5 Pakistani cities",
    statistics_config={"enabled": False},
)

try:
    engineered_fg.statistics_config = {"enabled": False}
    engineered_fg.update_statistics_config()
except Exception as e:
    print(f"Could not update statistics config (non-fatal): {e}")


cutoff_date = df_clean["date"].max() - pd.Timedelta(days=20)
df_to_push = df_clean[df_clean["date"] >= cutoff_date].copy()

print(f"Pushing recent window only: {len(df_to_push)} rows (was {len(df_clean)})")

expected_latest_date = df_clean["date"].max()
insert_succeeded = False

max_retries = 2
for attempt in range(1, max_retries + 1):
    try:
        engineered_fg.insert(df_to_push, write_options={"wait_for_job": False})
        print(f"Submitted {len(df_to_push)} rows to aqi_engineered_features "
              f"(not waiting for materialization job to finish).")
        insert_succeeded = True
        break
    except Exception as e:
        print(f"[Insert attempt {attempt}/{max_retries}] failed: {e}")
        if attempt < max_retries:
            wait = 15
            print(f"Waiting {wait}s before retry...")
            time.sleep(wait)

if not insert_succeeded:
    print("Insert retries failed. This run's engineered features were NOT pushed.")
    sys.exit(1)  # FAIL LOUDLY — don't hide this as success anymore

# ---- FRESHNESS CHECK ----
# The insert call succeeding doesn't guarantee Hopsworks' materialization
# job actually finished. Wait briefly, then re-read and confirm the FG
# actually reflects the new data before calling this run a success.
print("\nVerifying materialization actually landed...")
time.sleep(60)

try:
    check_df = engineered_fg.select_all().read(read_options={"use_hive": True})
    check_df["date"] = pd.to_datetime(check_df["date"])
    actual_latest = check_df["date"].max()
    print(f"Expected latest date: {expected_latest_date.date()}")
    print(f"Actual latest date in FG after materialization: {actual_latest.date()}")

    if actual_latest < expected_latest_date - pd.Timedelta(days=1):
        print("WARNING: Materialization appears stale/incomplete. "
              "Data was submitted but FG doesn't reflect it yet.")
        sys.exit(1)
    else:
        print("Materialization confirmed fresh.")
except Exception as e:
    print(f"Could not verify materialization (non-fatal, but suspicious): {e}")