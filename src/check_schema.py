import hopsworks
from dotenv import load_dotenv
import os

load_dotenv()

project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
fs = project.get_feature_store()

# Check the engineered feature group columns
fg = fs.get_feature_group("aqi_engineered_features", version=1)
df = fg.read()

print("=== COLUMNS IN aqi_engineered_features ===")
print(list(df.columns))
print()
print("=== SAMPLE ROW ===")
print(df.head(1).T)
print()

# Check each feature view's label column
for fv_name in ["aqi_fv_1d", "aqi_fv_2d", "aqi_fv_3d"]:
    fv = fs.get_feature_view(fv_name, version=1)
    print(f"=== {fv_name} schema ===")
    for feat in fv.schema:
        label_marker = " <-- LABEL" if feat.label else ""
        print(f"  {feat.name}{label_marker}")
    print()