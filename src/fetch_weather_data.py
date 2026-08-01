import requests
import pandas as pd


latitude = 31.5204
longitude = 74.3587


start_date = "2025-07-01"
end_date = "2026-07-01"



weather_url = "https://archive-api.open-meteo.com/v1/archive"


weather_params = {

    "latitude": latitude,

    "longitude": longitude,

    "start_date": start_date,

    "end_date": end_date,


    "hourly": [

        "temperature_2m",

        "relative_humidity_2m",

        "wind_speed_10m",

        "surface_pressure"

    ]

}


response = requests.get(
    weather_url,
    params=weather_params
)


data = response.json()


hourly = data["hourly"]


weather_df = pd.DataFrame({

    "time": hourly["time"],

    "temperature": hourly["temperature_2m"],

    "humidity": hourly["relative_humidity_2m"],

    "wind_speed": hourly["wind_speed_10m"],

    "pressure": hourly["surface_pressure"]

})


print(weather_df.head())


weather_df.to_csv(
    "data/historical_weather.csv",
    index=False
)


print("Weather data saved successfully")