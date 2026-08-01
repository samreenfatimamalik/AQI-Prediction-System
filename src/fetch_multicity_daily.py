import requests
import pandas as pd
import time

# Our 5 target cities with their coordinates
CITIES = {
    "Lahore": {"lat": 31.5204, "lon": 74.3587},
    "Karachi": {"lat": 24.8607, "lon": 67.0011},
    "Islamabad": {"lat": 33.6844, "lon": 73.0479},
    "Faisalabad": {"lat": 31.4504, "lon": 73.1350},
    "Peshawar": {"lat": 34.0151, "lon": 71.5249},
}

def fetch_with_retry(url, params, max_retries=3, timeout=60):
    """Try fetching data up to 3 times before giving up, with a 60-second timeout each try."""
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, params=params, timeout=timeout)
            response.raise_for_status()  # raises error if status code is bad (e.g. 404, 500)
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"  Attempt {attempt} failed: {e}")
            if attempt < max_retries:
                print(f"  Retrying in 5 seconds...")
                time.sleep(5)
            else:
                print(f"  Giving up after {max_retries} attempts.")
                raise  # re-raise the error so we know it truly failed


def fetch_city_data(city_name, lat, lon, start_date="2024-01-01", end_date="2026-07-24"):
    """Fetch historical hourly weather + AQI data for one city, then convert to daily average."""

    print(f"Fetching data for {city_name}...")

    # 1. Fetch air quality data (PM2.5, PM10)
    aqi_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    aqi_params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "pm2_5,pm10"
    }
    aqi_data = fetch_with_retry(aqi_url, aqi_params)

    # 2. Fetch weather data (temp, humidity, wind, pressure)
    weather_url = "https://archive-api.open-meteo.com/v1/archive"
    weather_params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,surface_pressure"
    }
    weather_data = fetch_with_retry(weather_url, weather_params)

    # 3. Convert both to pandas DataFrames
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

    # 5. Convert hourly -> DAILY AVERAGE
    df["date"] = df["time"].dt.date
    daily_df = df.groupby("date").agg({
        "pm2_5": "mean",
        "pm10": "mean",
        "temperature": "mean",
        "humidity": "mean",
        "wind_speed": "mean",
        "pressure": "mean"
    }).reset_index()

    # 6. Add city name column
    daily_df["city"] = city_name

    print(f"   Got {len(daily_df)} days of data for {city_name}")
    return daily_df


def fetch_all_cities():
    """Loop through all 5 cities and combine into one big dataset."""
    all_data = []

    for city_name, coords in CITIES.items():
        city_df = fetch_city_data(city_name, coords["lat"], coords["lon"])
        all_data.append(city_df)
        time.sleep(2)  # Be polite to the API, avoid rate limits

    # Combine all cities into one DataFrame
    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df


if __name__ == "__main__":
    final_df = fetch_all_cities()
    print(f"\n Done! Total rows: {len(final_df)}")
    print(final_df.head(10))

    # Save locally first, just to check it looks right
    final_df.to_csv("data/multicity_daily_aqi.csv", index=False)
    print("\n Saved to data/multicity_daily_aqi.csv")