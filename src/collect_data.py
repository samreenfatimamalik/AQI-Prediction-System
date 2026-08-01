#print("hello")
import requests

#APIs generally require goegraphicsl coodinates(longitude and latitude) to provide location, by using this we access lahore

latitude = 31.5204
longitude = 74.3587
url = (
    "https://air-quality-api.open-meteo.com/v1/air-quality?"
    f"latitude={latitude}"
    f"&longitude={longitude}"
    "&current=pm10,pm2_5"
)
#print(url)
response = requests.get(url)

#print(response)
#print(response.status_code)
data = response.json()
#print(data)
current_data = data["current"]


print("AQI Data Collection")
print("-------------------")

print("Time:", current_data["time"])
print("PM10:", current_data["pm10"])
print("PM2.5:", current_data["pm2_5"])