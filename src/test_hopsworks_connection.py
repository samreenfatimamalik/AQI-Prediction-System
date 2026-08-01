import os
import hopsworks
from dotenv import load_dotenv

# Load the API key from .env file
load_dotenv()

# Connect to Hopsworks
project = hopsworks.login(
    api_key_value=os.getenv("HOPSWORKS_API_KEY"),
    project="my_aqi_predictor"  #  project name
)

print("Connected successfully to project:", project.name)