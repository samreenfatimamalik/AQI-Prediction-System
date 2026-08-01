import os
import pandas as pd
import hopsworks
from dotenv import load_dotenv

load_dotenv()

# 1. Connect to Hopsworks
print("Connecting to Hopsworks...")
project = hopsworks.login(
    api_key_value=os.getenv("HOPSWORKS_API_KEY"),
    project="my_aqi_predictor"  #  actual project name
)

# 2. Get the Feature Store (this is where our data will live)
fs = project.get_feature_store()

# 3. Load our local CSV
df = pd.read_csv("data/multicity_daily_aqi.csv")
df["date"] = pd.to_datetime(df["date"])  # make sure it's a proper date type

print(f"Loaded {len(df)} rows, preparing to upload...")

aqi_fg = fs.get_or_create_feature_group(
    name="aqi_daily_features",
    version=1,
    description="Daily average AQI and weather data for 5 Pakistani cities",
    primary_key=["city", "date"],   # uniquely identifies each row
    event_time="date",              # tells Hopsworks this is time-series data
    time_travel_format="HUDI"       #  HUDI maintains history of data
)


# 5. Insert our data
aqi_fg.insert(df)

print(" Data successfully pushed to Hopsworks Feature Store!")