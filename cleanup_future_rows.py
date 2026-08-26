import hopsworks
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
fs = project.get_feature_store()

today = pd.Timestamp.now(tz="UTC").normalize()
print(f"Treating anything after {today.date()} as a bad future-dated row.\n")

for fg_name in ["aqi_daily_features", "aqi_engineered_features"]:
    print(f"--- Checking {fg_name} ---")
    fg = fs.get_feature_group(fg_name, version=1)
    df = fg.read()
    df["date"] = pd.to_datetime(df["date"])

    future_rows = df[df["date"] > today]
    clean_df = df[df["date"] <= today]

    print(f"Found {len(future_rows)} future-dated rows out of {len(df)} total.")
    print(f"Clean data will have {len(clean_df)} rows.")

    if len(future_rows) > 0:
        confirm = input(f"Overwrite {fg_name} with cleaned data (removing {len(future_rows)} rows)? (y/n): ")
        if confirm.lower() == "y":
            fg.insert(clean_df, overwrite=True, write_options={"wait_for_job": True})
            print(f"{fg_name} overwritten with clean data.\n")
        else:
            print("Skipped.\n")
    else:
        print("Nothing to clean.\n")

print("Cleanup complete.")