#import hopsworks
#from dotenv import load_dotenv
#import os

#load_dotenv()
#project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
#fs = project.get_feature_store()
#fg = fs.get_feature_group("aqi_daily_features", version=1)

#df = fg.select_all().read(read_options={"use_hive": True})
#print("Row count (Hive engine):", len(df))

import hopsworks
from dotenv import load_dotenv
import os

load_dotenv()
project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
fs = project.get_feature_store()
fg = fs.get_feature_group("aqi_engineered_features", version=1)
df = fg.select_all().read(read_options={"use_hive": True})
print("Row count:", len(df))