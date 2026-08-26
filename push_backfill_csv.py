import pandas as pd
import hopsworks
import os
from dotenv import load_dotenv

load_dotenv()

df = pd.read_csv("data/multicity_daily_aqi.csv")
df["date"] = pd.to_datetime(df["date"])

print(f"Loaded {len(df)} rows from CSV.")
print(df.head())

project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
fs = project.get_feature_store()

fg = fs.get_feature_group("aqi_daily_features", version=1)

fg.insert(df, write_options={"wait_for_job": True})
print(f"\nSuccessfully pushed {len(df)} rows to aqi_daily_features.")