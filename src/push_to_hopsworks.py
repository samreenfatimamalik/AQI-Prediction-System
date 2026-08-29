import requests
import pandas as pd
import time
import os
import hopsworks
from dotenv import load_dotenv

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

# Minimum days to always re-fetch, even if no gap detected
# (catches today's data filling in hour by hour)
MIN_PAST_DAYS = 3

# Safety cap — don't ever try to fetch more than this many days back
# (Open-Meteo forecast API supports up to ~92 days of past_days)
MAX_PAST_DAYS = 92


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


def get_last_inserted_date(fg, city_name):
    """Check Hopsworks for the most recent date already stored for this city.
    Returns None if the city has no data yet (first-ever run)."""
    try:
        df = fg.read()
        city_df = df[df["city"] == city_name]
        if city_df.empty:
            return None
        last_date = pd.to_datetime(city_df["date"]).max().normalize()
        if last_date.tz is not None:
            last_date = last_date.tz_localize(None)   # <-- yeh line add ki
        return last_date
    except Exception as e:
        print(f"  Could not check last date for {city_name} (probably first run): {e}")
        return None


def compute_past_days_needed(last_date):
    """Work out how many days back we need to fetch to close the gap,
    with a sensible minimum and a safety cap."""
    if last_date is None:
        return MIN_PAST_DAYS

    today = pd.Timestamp.now().normalize()
    gap_days = (today - last_date).days

    # Always fetch at least MIN_PAST_DAYS (to refresh today's partial data),
    # but if the gap is bigger, fetch enough to cover it, capped at MAX_PAST_DAYS.
    needed = max(MIN_PAST_DAYS, gap_days + 1)
    return min(needed, MAX_PAST_DAYS)


def fetch_city_latest(city_name, lat, lon, past_days):
    """Fetch the last N days of hourly weather + AQI data for one city, then convert to daily average."""

    print(f"Fetching latest data for {city_name} (past_days={past_days})...")

    # 1. Air quality data
    aqi_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    aqi_params = {
        "latitude": lat,
        "longitude": lon,
        "past_days": past_days,
        "hourly": "pm2_5,pm10"
    }
    aqi_data = fetch_with_retry(aqi_url, aqi_params)

    # 2. Weather data (forecast API, for near-real-time recent data)
    weather_url = "https://api.open-meteo.com/v1/forecast"
    weather_params = {
        "latitude": lat,
        "longitude": lon,
        "past_days": past_days,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,surface_pressure"
    }
    weather_data = fetch_with_retry(weather_url, weather_params)

    # 3. Convert to DataFrames
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

    # 5. Drop any future/forecasted hours — we only want data that actually happened
    today = pd.Timestamp.now().normalize()
    df = df[df["time"] < today + pd.Timedelta(days=1)]

    # 6. Hourly -> daily average
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


def fetch_all_cities_latest(aqi_fg):
    """Loop through all 5 cities. For each city, check how far behind Hopsworks is,
    then fetch exactly enough days to close that gap."""
    all_data = []
    for city_name, coords in CITIES.items():
        last_date = get_last_inserted_date(aqi_fg, city_name)
        past_days = compute_past_days_needed(last_date)

        if last_date is not None:
            print(f"{city_name}: last data in Hopsworks is {last_date.date()}, fetching past_days={past_days}")
        else:
            print(f"{city_name}: no existing data found, fetching default past_days={past_days}")

        city_df = fetch_city_latest(city_name, coords["lat"], coords["lon"], past_days=past_days)
        all_data.append(city_df)
        time.sleep(2)  # be polite to the API

    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df


def get_feature_group(project):
    fs = project.get_feature_store()
    aqi_fg = fs.get_or_create_feature_group(
        name="aqi_daily_features",
        version=1,
        description="Daily average AQI and weather data for 5 Pakistani cities",
        primary_key=["city", "date"],
        event_time="date",
        time_travel_format="HUDI"
    )
    return aqi_fg


def push_to_hopsworks(df, aqi_fg):
    """Push the batch into the feature group."""
    df["date"] = pd.to_datetime(df["date"])
    
    # Check if a previous materialization job is still running
    try:
        job_state = aqi_fg.materialization_job.get_state()
        if job_state in ("RUNNING", "INITIALIZING"):
            print(f"Materialization job already running (state={job_state}). "
                  "Skipping insert this run to avoid overlap — next scheduled run will retry.")
            return
    except Exception as e:
        print(f"Could not check materialization job state (continuing anyway): {e}")
    
    insert_with_retry(aqi_fg, df)
    print("Latest data successfully upserted into Hopsworks!")


if __name__ == "__main__":
    print("Connecting to Hopsworks...")
    project = hopsworks.login(
        api_key_value=os.getenv("HOPSWORKS_API_KEY"),
        project="my_aqi_predictor"
    )
    aqi_fg = get_feature_group(project)

    latest_df = fetch_all_cities_latest(aqi_fg)
    print(f"\nFetched {len(latest_df)} rows across all cities.")
    print(latest_df.head(10))

    push_to_hopsworks(latest_df, aqi_fg)