import requests
import pandas as pd


latitude = 31.5204
longitude = 74.3587


start_date = "2025-07-01"
end_date = "2026-07-01"


aqi_url = "https://air-quality-api.open-meteo.com/v1/air-quality"


aqi_params = {

    "latitude": latitude,
    "longitude": longitude,

    "start_date": start_date,
    "end_date": end_date,

    "hourly": [
        "pm10",
        "pm2_5"
    ]

}


response = requests.get(
    aqi_url,
    params=aqi_params
)


data = response.json()


hourly = data["hourly"]


aqi_df = pd.DataFrame({

    "time": hourly["time"],

    "PM10": hourly["pm10"],

    "PM2.5": hourly["pm2_5"]

})


print(aqi_df.head())


aqi_df.to_csv(
    "data/historical_aqi.csv",
    index=False
)


print("AQI data saved successfully")