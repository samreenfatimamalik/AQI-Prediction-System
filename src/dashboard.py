import os
import glob
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import shap
import hopsworks
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Pearls AQI Predictor", page_icon="🌫️", layout="wide")
st.markdown("""
<style>
.block-container { padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)

CITIES = ["Lahore", "Karachi", "Islamabad", "Faisalabad", "Peshawar"]

st.markdown("""
<style>
.metric-card {
    background-color: #1e2530; border-radius: 12px; padding: 20px;
    text-align: center; border: 1px solid #2d3748;
}
.metric-card h1 { margin: 5px 0; font-size: 42px; }
.metric-card p { margin: 0; color: #9ca3af; font-size: 14px; }
</style>
""", unsafe_allow_html=True)


def get_aqi_category(value):
    if value <= 50:
        return "Good", "#2ecc71"
    elif value <= 100:
        return "Moderate", "#f1c40f"
    elif value <= 150:
        return "Unhealthy for Sensitive Groups", "#e67e22"
    elif value <= 200:
        return "Unhealthy", "#e74c3c"
    elif value <= 300:
        return "Very Unhealthy", "#8e44ad"
    else:
        return "Hazardous", "#7d3c3c"


@st.cache_resource(show_spinner="Connecting to Hopsworks...")
def get_project():
    return hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))


@st.cache_resource(show_spinner="Loading models from registry...")
def load_models():
    project = get_project()
    mr = project.get_model_registry()
    models = {}
    for horizon in ["1d", "2d", "3d"]:
        m = mr.get_model(f"aqi_model_{horizon}", version=1)
        model_dir = m.download()
        pkl_files = glob.glob(os.path.join(model_dir, "*.pkl"))
        models[horizon] = joblib.load(pkl_files[0])
    return models


@st.cache_data(ttl=3600, show_spinner="Fetching latest data from Hopsworks...")
def load_all_features():
    project = get_project()
    fs = project.get_feature_store()
    fg = fs.get_feature_group("aqi_engineered_features", version=1)
    try:
        df = fg.read()
    except Exception:
        df = fg.select_all().read(read_options={"use_hive": True})
    df["date"] = pd.to_datetime(df["date"])

    # Defensive guard: never trust future-dated rows, even if some slip
    # into the feature store (e.g. from an API returning forecast data).
    today = pd.Timestamp.now(tz="UTC").normalize()
    df = df[df["date"] <= today]

    cutoff = df["date"].max() - pd.Timedelta(days=15)
    return df[df["date"] >= cutoff].reset_index(drop=True)


def build_engineered_features(df):
    """Works on single-row or multi-row DataFrames alike."""
    df = df.copy()
    df["aqi_change_rate"] = (df["pm25_lag_1"] - df["pm25_lag_3"]) / 3
    df["volatility_7"] = (df["pm25_lag_1"] - df["pm25_rolling_7"]).abs()
    df["wind_stagnation"] = df["humidity"] / (df["wind_speed"] + 1)
    df["pressure_humidity"] = df["pressure"] * df["humidity"] / 1000
    df["pm25_trend"] = df["pm25_lag_1"] - df["pm25_lag_7"]
    df["pm25_rolling_max_7"] = df[
        ["pm25_lag_1", "pm25_lag_3", "pm25_lag_7", "pm25_rolling_7"]
    ].max(axis=1)

    dummies = pd.get_dummies(df["city"], prefix="city")
    for c in CITIES:
        col = f"city_{c}"
        if col not in dummies.columns:
            dummies[col] = 0
    return pd.concat([df, dummies], axis=1)


@st.cache_data(show_spinner=False)
def get_engineered_all(all_data):
    return build_engineered_features(all_data)


def predict_for_city(city_df, models):
    latest_row = city_df.sort_values("date", ascending=False).iloc[[0]]
    latest_row = build_engineered_features(latest_row)
    predictions = {}
    for horizon, model in models.items():
        feature_cols = list(model.feature_names_in_)
        predictions[horizon] = model.predict(latest_row[feature_cols])[0]
    return predictions, latest_row


def predict_all_cities(all_data, models):
    rows = []
    for city in CITIES:
        city_df = all_data[all_data["city"] == city]
        if city_df.empty:
            continue
        preds, _ = predict_for_city(city_df, models)
        cat, _ = get_aqi_category(preds["1d"])
        rows.append({
            "City": city, "Tomorrow": round(preds["1d"]),
            "In 2 days": round(preds["2d"]), "In 3 days": round(preds["3d"]),
            "Status": cat,
        })
    return pd.DataFrame(rows)


@st.cache_resource(show_spinner=False)
def get_shap_explainer(horizon_key, _model, _background_df):
    return shap.Explainer(_model, _background_df)


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("🌫️ Pearls AQI")
    st.caption("Serverless 3-day AQI forecaster")
    city = st.selectbox("Select a city", CITIES)
    if st.button("🔄 Refresh data"):
        st.cache_data.clear()
        st.rerun()

# ---------------- LOAD DATA ----------------
try:
    all_data = load_all_features()
    models = load_models()
    city_df = all_data[all_data["city"] == city]

    if city_df.empty:
        st.warning(f"No data found for {city} yet.")
        st.stop()

    predictions, latest_row = predict_for_city(city_df, models)
    last_date = latest_row["date"].iloc[0]

    with st.sidebar:
        st.divider()
        st.caption(f"Data as of {last_date.strftime('%Y-%m-%d')}")
        st.metric("Temperature", f"{latest_row['temperature'].iloc[0]:.1f}°C")
        st.metric("Humidity", f"{latest_row['humidity'].iloc[0]:.0f}%")
        st.metric("Wind Speed", f"{latest_row['wind_speed'].iloc[0]:.1f} km/h")
        st.metric("Pressure", f"{latest_row['pressure'].iloc[0]:.0f} hPa")

    # ---------------- HEADER ----------------
    st.title("Pearls AQI Predictor")
    st.caption(f"3-day AQI forecast for {city}, Pakistan")

    cols = st.columns(3)
    label_dates = {
        "1d": last_date + pd.Timedelta(days=1),
        "2d": last_date + pd.Timedelta(days=2),
        "3d": last_date + pd.Timedelta(days=3),
    }
    labels = {h: d.strftime("%a, %b %d") for h, d in label_dates.items()}  # e.g. "Tue, Aug 11"
    hazard = False
    for i, h in enumerate(["1d", "2d", "3d"]):
        val = predictions[h]
        cat, color = get_aqi_category(val)
        if val > 150:
            hazard = True
        with cols[i]:
            st.markdown(f"""
            <div class="metric-card">
                <p>{labels[h]}</p>
                <h1 style="color:{color}">{val:.0f}</h1>
                <p style="color:{color}; font-weight:600">{cat}</p>
            </div>
            """, unsafe_allow_html=True)

    st.write("")
    if hazard:
        st.error(f"⚠️ Hazardous AQI levels predicted for {city}. Sensitive groups should limit outdoor exposure.")
    else:
        st.success(f"No hazardous AQI levels predicted for {city} in the next 3 days.")

    st.write("")
    tab1, tab2, tab3 = st.tabs(["📈 Trend History", "🏙️ Compare Cities", "🔍 Why this prediction?"])

    # ---------------- TAB 1: TREND ----------------
    with tab1:
        hist = city_df.sort_values("date")
        forecast_dates = [last_date + pd.Timedelta(days=d) for d in [1, 2, 3]]
        forecast_vals = [predictions["1d"], predictions["2d"], predictions["3d"]]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hist["date"], y=hist["pm2_5"], mode="lines+markers",
            name="Observed PM2.5", line=dict(color="#3498db")
        ))
        fig.add_trace(go.Scatter(
            x=forecast_dates, y=forecast_vals, mode="lines+markers",
            name="Forecast", line=dict(color="#e74c3c", dash="dash")
        ))
        fig.update_layout(
            template="plotly_dark", height=420,
            xaxis_title="Date", yaxis_title="PM2.5 / AQI",
            legend=dict(orientation="h", y=1.1),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ---------------- TAB 2: COMPARE CITIES ----------------
    with tab2:
        comp_df = predict_all_cities(all_data, models)
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

        fig2 = px.bar(
            comp_df.melt(id_vars=["City", "Status"], value_vars=["Tomorrow", "In 2 days", "In 3 days"],
                          var_name="Horizon", value_name="AQI"),
            x="City", y="AQI", color="Horizon", barmode="group",
            template="plotly_dark", height=420,
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ---------------- TAB 3: SHAP ----------------
    with tab3:
        horizon_choice = st.radio("Explain which forecast?", ["1d", "2d", "3d"],
                                   format_func=lambda h: labels[h], horizontal=True)
        model = models[horizon_choice]
        feature_cols = list(model.feature_names_in_)

        with st.spinner("Computing SHAP explanation..."):
            engineered_all = get_engineered_all(all_data)
            background = engineered_all[feature_cols].sample(
                min(20, len(engineered_all)), random_state=42
            )
            explainer = get_shap_explainer(horizon_choice, model, background)
            row_for_shap = latest_row[feature_cols]
            explanation = explainer(row_for_shap)

        values = explanation.values[0]
        shap_df = pd.DataFrame({
            "feature": feature_cols, "impact": values
        }).sort_values("impact", key=abs, ascending=True).tail(10)
        shap_df["color"] = shap_df["impact"].apply(lambda v: "#e74c3c" if v > 0 else "#2ecc71")

        fig3 = go.Figure(go.Bar(
            x=shap_df["impact"], y=shap_df["feature"], orientation="h",
            marker_color=shap_df["color"]
        ))
        fig3.update_layout(
            template="plotly_dark", height=420,
            xaxis_title="Impact on predicted AQI",
            title=f"Top feature contributions — {labels[horizon_choice]} forecast for {city}",
        )
        st.plotly_chart(fig3, use_container_width=True)
        st.caption("🔴 Red bars push the prediction higher (worse air quality). 🟢 Green bars push it lower.")

except Exception as e:
    st.error(f"Something went wrong: {e}")
    st.info("Check that your .env file has HOPSWORKS_API_KEY set correctly.")