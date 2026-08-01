import pandas as pd


# Load AQI data
aqi_df = pd.read_csv(
    "data/historical_aqi.csv"
)


# Load Weather data
weather_df = pd.read_csv(
    "data/historical_weather.csv"
)


# Merge both datasets using time column
final_df = pd.merge(

    aqi_df,

    weather_df,

    on="time",

    how="inner"

)


# Display merged data
print(final_df.head())


# Check columns
print(final_df.info())


# Save final dataset
final_df.to_csv(

    "data/historical_data.csv",

    index=False

)


print("Final merged dataset saved successfully!")