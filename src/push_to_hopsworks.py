import requests
import pandas as pd
import time
import os
import hopsworks
from dotenv import load_dotenv
import time

def insert_with_retry(fg, df, max_retries=3, wait_seconds=20, **kwargs):
    for attempt in range(1, max_retries + 1):
        try:
            fg.insert(df, **kwargs)
            return
        except Exception as e:
            print(f"[Insert attempt {attempt}/{max_retries}] failed: {e}")
            if attempt < max_retries:
                print(f"Waiting {wait_seconds}s before retry...")
                time.sleep(wait_seconds)
            else:
                print("All insert retries failed. Hopsworks appears unresponsive right now. "
                      "Skipping this run — next scheduled run will try again.")
                raise

load_dotenv()

# Same 5 cities as before
CITIES = {
    "Lahore": {"lat": 31.5204, "lon": 74.3587},
    "Karachi": {"lat": 24.8607, "lon": 67.0011},
    "Islamabad": {"lat": 33.6844, "lon": 73.0479},
    "Faisalabad": {"lat": 31.4504, "lon": 73.1350},
    "Peshawar": {"lat": 34.0151, "lon": 71.5249},
}

# How many past days to re-fetch each run.
# We re-fetch a small window (not just "today") to catch:
#   1) today's data filling in hour by hour (it's incomplete until the day ends)
#   2) any hours the API hadn't published yet on the last run
PAST_DAYS = 3


def fetch_with_retry(url, params, max_retries=3, timeout=60):
    """Try fetching data up to 3 times before giving up, with a 60-second timeout each try."""
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"  Attempt {attempt} failed: {e}")
            if attempt < max_retries:
                print(f"  Retrying in 5 seconds...")
                time.sleep(5)
            else:
                print(f"  Giving up after {max_retries} attempts.")
                raise


def fetch_city_latest(city_name, lat, lon, past_days=PAST_DAYS):
    """Fetch the last few days of hourly weather + AQI data for one city, then convert to daily average."""

    print(f"Fetching latest data for {city_name}...")

    # 1. Air quality data - this API is already real-time, so past_days works directly
    aqi_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    aqi_params = {
        "latitude": lat,
        "longitude": lon,
        "past_days": past_days,
        "hourly": "pm2_5,pm10"
    }
    aqi_data = fetch_with_retry(aqi_url, aqi_params)

    # 2. Weather data - using the FORECAST api (not archive!) so we get near-real-time recent data
    weather_url = "https://api.open-meteo.com/v1/forecast"
    weather_params = {
        "latitude": lat,
        "longitude": lon,
        "past_days": past_days,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,surface_pressure"
    }
    weather_data = fetch_with_retry(weather_url, weather_params)

    # 3. Convert to DataFrames (same as your original script)
    df_aqi = pd.DataFrame({
        "time": aqi_data["hourly"]["time"],
        "pm2_5": aqi_data["hourly"]["pm2_5"],
        "pm10": aqi_data["hourly"]["pm10"]
    })

    df_weather = pd.DataFrame({
        "time": weather_data["hourly"]["time"],
        "temperature": weather_data["hourly"]["temperature_2m"],
        "humidity": weather_data["hourly"]["relative_humidity_2m"],
        "wind_speed": weather_data["hourly"]["wind_speed_10m"],
        "pressure": weather_data["hourly"]["surface_pressure"]
    })


   # 4. Merge on time
    df = pd.merge(df_aqi, df_weather, on="time")
    df["time"] = pd.to_datetime(df["time"])

    # 5. Drop any future/forecasted hours — Open-Meteo's forecast API
    #    returns upcoming days by default, which are PREDICTIONS, not
    #    real observed pollution. We only want data that has actually happened.
    today = pd.Timestamp.now().normalize()
    df = df[df["time"] < today + pd.Timedelta(days=1)]
    # 6. Hourly -> daily average
    df["date"] = df["time"].dt.date

    # 5. Hourly -> daily average
    df["date"] = df["time"].dt.date
    daily_df = df.groupby("date").agg({
        "pm2_5": "mean",
        "pm10": "mean",
        "temperature": "mean",
        "humidity": "mean",
        "wind_speed": "mean",
        "pressure": "mean"
    }).reset_index()

    daily_df["city"] = city_name

    print(f"   Got {len(daily_df)} days of data for {city_name}")
    return daily_df


def fetch_all_cities_latest():
    """Loop through all 5 cities and combine into one small recent dataset."""
    all_data = []
    for city_name, coords in CITIES.items():
        city_df = fetch_city_latest(city_name, coords["lat"], coords["lon"])
        all_data.append(city_df)
        time.sleep(2)  # be polite to the API

    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df


def push_to_hopsworks(df):
    """Push the small recent batch into the SAME feature group as before.
    Because primary_key=['city','date'] is set, Hopsworks/HUDI will UPSERT:
    matching rows get updated in place, no duplicates get created."""

    print("Connecting to Hopsworks...")
    project = hopsworks.login(
        api_key_value=os.getenv("HOPSWORKS_API_KEY"),
        project="my_aqi_predictor"
    )
    fs = project.get_feature_store()

    df["date"] = pd.to_datetime(df["date"])

    aqi_fg = fs.get_or_create_feature_group(
        name="aqi_daily_features",
        version=1,
        description="Daily average AQI and weather data for 5 Pakistani cities",
        primary_key=["city", "date"],
        event_time="date",
        time_travel_format="HUDI"
    )

    insert_with_retry(aqi_fg, df)
    print("Latest data successfully upserted into Hopsworks!")


if __name__ == "__main__":
    latest_df = fetch_all_cities_latest()
    print(f"\nFetched {len(latest_df)} rows across all cities (last {PAST_DAYS} days).")
    print(latest_df.head(10))

    push_to_hopsworks(latest_df)