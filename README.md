# AQI Predictor

A serverless, end-to-end machine learning system that forecasts the Air Quality Index (AQI) for the next three days across five major Pakistani cities: Lahore, Karachi, Islamabad, Faisalabad, and Peshawar.

The system automatically collects weather and pollution data, engineers features, trains machine learning models, and serves predictions through an interactive dashboard, without any manual intervention after deployment.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Live Demo](#live-demo)
3. [System Architecture](#system-architecture)
4. [Technology Stack](#technology-stack)
5. [Project Structure](#project-structure)
6. [How the Pipeline Works](#how-the-pipeline-works)
7. [Model Performance](#model-performance)
8. [Dashboard Features](#dashboard-features)
9. [Automation (CI/CD)](#automation-cicd)
10. [Handling a Free-Tier Feature Store](#handling-a-free-tier-feature-store)
11. [Local Setup](#local-setup)
12. [Environment Variables](#environment-variables)
13. [Known Limitations](#known-limitations)
14. [Future Improvements](#future-improvements)

---

## Project Overview

Air pollution in Pakistani cities changes quickly due to traffic, weather, seasonal smog, dust storms, and crop burning. This project builds a complete forecasting pipeline that:

- Collects real-time and historical weather and air quality data from a public API
- Stores the data in a centralized feature store instead of local files
- Engineers time-based and pollution-based features for forecasting
- Trains separate models to predict AQI 1, 2, and 3 days ahead
- Runs automatically on a schedule, with no manual data entry or retraining
- Explains each prediction using SHAP feature importance
- Displays results on a live dashboard with hazardous AQI alerts

The entire system runs without a traditional backend server. Data collection and model training happen through scheduled automation, and the dashboard reads directly from the feature store and model registry.

---

## Live Demo

- Dashboard: https://sam-aqi-predictor.streamlit.app/
- Repository: https://github.com/samreenfatimamalik/AQI-Prediction-System

---

## System Architecture

```
                        Open-Meteo API
                              |
                              v
                 GitHub Actions (every 4 hours)
                              |
              ---------------------------------
              |                               |
     Raw feature pipeline           Feature engineering pipeline
     (push_to_hopsworks.py)         (fetch_from_hopsworks.py)
              |                               |
              v                               v
        Hopsworks Feature Store (aqi_daily_features, aqi_engineered_features)
                              |
                              v
                 GitHub Actions (daily, 02:00)
                              |
                              v
              Model training (1-day, 2-day, 3-day models)
                              |
                              v
                 Hopsworks Model Registry
                              |
                              v
                 Streamlit Dashboard (reads live from
                 Feature Store + Model Registry)
```

There is no dedicated backend server. The dashboard itself connects to Hopsworks at runtime, loads the latest features and the latest trained models, and computes predictions on the fly whenever a user opens the app.

---

## Technology Stack

| Category | Tools Used |
|---|---|
| Language | Python |
| Data Source | Open-Meteo Air Quality API and Open-Meteo Weather API (no API key required) |
| Feature Store & Model Registry | Hopsworks |
| Machine Learning | Scikit-learn (Linear Regression, Random Forest, HistGradientBoosting) |
| Explainability | SHAP |
| Automation / CI-CD | GitHub Actions |
| Dashboard | Streamlit |
| Visualization | Plotly |
| Version Control | Git and GitHub |

---

## Project Structure

```
AQI-Prediction-System/
│
├── .github/
│   └── workflows/
│       ├── hourly_feature_pipeline.yml     # Data collection, every 4 hours
│       └── daily_training_pipeline.yml     # Model retraining, once a day
│
├── data/                     # Local data checkpoints (CSV snapshots)
├── models/                   # Locally cached model artifacts
├── notebooks/                # Exploratory data analysis
│
├── src/
│   ├── push_to_hopsworks.py       # Fetches raw weather/AQI data and pushes it to Hopsworks
│   ├── fetch_from_hopsworks.py    # Reads raw data, engineers features, pushes engineered data
│   ├── train_model_1d.py          # Trains and registers the 1-day forecasting model
│   ├── train_model_2d.py          # Trains and registers the 2-day forecasting model
│   ├── train_model_3d.py          # Trains and registers the 3-day forecasting model
│   ├── fetch_multicity_daily.py   # Historical backfill script (one-time use)
│   └── dashboard.py               # Streamlit application
│
└── README.md
```

---

## How the Pipeline Works

### 1. Data Collection

`push_to_hopsworks.py` fetches hourly weather and air quality data for all five cities from the Open-Meteo API, converts it into daily averages, and inserts it into the `aqi_daily_features` feature group in Hopsworks.

The script checks the most recent date already stored in Hopsworks for each city and automatically fetches only the missing days since that date. This means that even if a scheduled run fails, the next successful run automatically backfills the gap, and no manual intervention is required to keep the dataset complete.

### 2. Feature Engineering

`fetch_from_hopsworks.py` reads the raw daily data, sorts it by city and date, and creates the following features:

- Time-based features: day of week, month
- Lag features: PM2.5 value 1, 3, and 7 days earlier
- Rolling features: 3-day and 7-day rolling averages of PM2.5
- Prediction targets: PM2.5 value 1, 2, and 3 days ahead

All lag and rolling calculations are grouped by city to avoid mixing data between cities. The most recent rows (which do not yet have a target value because the future has not occurred yet) are intentionally kept in the feature group so the dashboard can use them for live inference.

The engineered features are then pushed to a second feature group, `aqi_engineered_features`.

### 3. Model Training

Three separate models are trained, one for each forecasting horizon:

- `aqi_model_1d` — predicts PM2.5 one day ahead
- `aqi_model_2d` — predicts PM2.5 two days ahead
- `aqi_model_3d` — predicts PM2.5 three days ahead

Each model is trained using a time-based split (older data for training, newer data for testing) rather than a random split, since shuffling would leak future information into training. Models are evaluated using MAE, RMSE, and R², and the best-performing model for each horizon is registered in the Hopsworks Model Registry.

### 4. Prediction and Explanation

The dashboard loads the latest engineered features and the three registered models, generates predictions for the next three days, and uses SHAP to explain which features contributed most to each prediction.

---

## Model Performance

| Horizon | Best Model | R² Score |
|---|---|---|
| 1-day ahead | Linear Regression | 0.65 |
| 2-day ahead | HistGradientBoosting | 0.51 |
| 3-day ahead | HistGradientBoosting | 0.50 |

Performance decreases with longer forecasting horizons. This is expected because the model does not have access to actual future weather forecasts for the target date; it can only rely on lag and rolling features from past days. Karachi is the most predictable city due to relatively stable coastal weather, while Lahore and Faisalabad show more error due to sudden pollution spikes caused by events such as crop burning and dust storms, which are not captured by the available features.

---

## Dashboard Features

The dashboard is built with Streamlit and is styled as an environmental sensor console. It includes:

- A city selector for all five cities
- A live AQI reading with a color-coded status (Good, Moderate, Unhealthy, Very Unhealthy, Hazardous)
- A 3-day forecast panel showing predicted AQI for each of the next three days
- A hazardous AQI alert banner that activates automatically when any forecasted value crosses the hazardous threshold
- A historical trend chart comparing observed PM2.5 values with forecasted values
- A city comparison view showing predictions for all five cities side by side
- A "Why this prediction" tab that uses SHAP to show which features increased or decreased the forecasted AQI
- A manual refresh option to force a fresh read from the feature store

---

## Automation (CI/CD)

Two GitHub Actions workflows keep the system running without manual work:

**Hourly Feature Pipeline** (runs every 4 hours)
1. Fetches the latest raw weather and AQI data and pushes it to Hopsworks
2. Waits briefly for the write to be committed
3. Reads the raw data, engineers features, and pushes the engineered feature set

**Daily Training Pipeline** (runs once a day)
1. Retrieves the latest engineered features from Hopsworks
2. Retrains all three models (1-day, 2-day, 3-day)
3. Registers the updated models in the Model Registry

Both workflows use `workflow_dispatch`, so they can also be triggered manually from the GitHub Actions tab for testing.

---

## Handling a Free-Tier Feature Store

Hopsworks' free tier introduces server-side instability that is not related to application code, including transient connection drops when reading data, and slow or occasionally failed background jobs that commit data to storage. Since this is a real constraint of building a production-style pipeline on free infrastructure, the pipeline is designed to be resilient to it rather than to assume perfect uptime:

- **Retry logic**: Every read and write operation retries automatically before failing, and falls back to an alternative read path if the primary one is unavailable.
- **Gap detection and backfill**: Before fetching new data, the pipeline checks the last date already stored for each city and fetches exactly the missing range, so a failed run does not create a permanent hole in the dataset.
- **Non-blocking writes**: Feature inserts do not block the pipeline while waiting for background storage jobs to finish, since those jobs can be slow on shared free-tier infrastructure. The pipeline moves on, and the next scheduled run naturally reconciles any pending data.
- **Conflict avoidance**: Before writing new data, the pipeline checks whether a previous background job is still in progress and skips the write if so, rather than triggering a conflicting job.
- **Graceful degradation**: If Hopsworks is completely unreachable during a run, the pipeline exits cleanly and logs the reason instead of crashing, and the next scheduled run retries automatically.
This design means the system keeps itself consistent over time even when individual runs occasionally fail, which is expected behavior on free-tier infrastructure.

---

## Local Setup

### Prerequisites

- Python 3.11
- A Hopsworks account and project
- Git

### Steps

```bash
# Clone the repository
git clone https://github.com/samreenfatimamalik/AQI-Prediction-System.git
cd AQI-Prediction-System

# Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # Windows
source .venv/bin/activate     # macOS/Linux

# Install dependencies
pip install requests pandas hopsworks python-dotenv pyarrow scikit-learn streamlit plotly shap joblib
```

Create a `.env` file in the project root:


Run the pipeline manually:

```bash
python src/push_to_hopsworks.py
python src/fetch_from_hopsworks.py
python src/train_model_1d.py
python src/train_model_2d.py
python src/train_model_3d.py
```

Run the dashboard:

```bash
streamlit run src/dashboard.py
```

---

## Environment Variables

| Variable | Description |
|---|---|
| `HOPSWORKS_API_KEY` | API key used to authenticate with the Hopsworks project. Required locally (via `.env`), in GitHub Actions (as a repository secret), and in Streamlit Cloud (as an app secret). |

---

## Known Limitations

- Forecast accuracy decreases for the 2-day and 3-day horizons because the model has no access to actual forecasted weather for future dates, only past observations.
- Sudden, event-driven pollution spikes (crop burning, dust storms) are difficult to predict without additional data sources such as satellite fire detection or dedicated weather forecasts.
- The free tier of the feature store introduces occasional transient failures, which the pipeline is designed to recover from automatically rather than prevent entirely.

---

## Future Improvements

- Incorporate actual weather forecast data (rather than only historical weather) to improve 2-day and 3-day accuracy
- Add satellite-based fire and dust detection as an additional feature source
- Experiment with deep learning models (LSTM or similar sequence models) for longer horizons
- Add automated model comparison and rollback if a newly trained model performs worse than the currently deployed one
- Extend coverage to additional cities

## BTW I am still working on this, Stay updated for more ℹ️
