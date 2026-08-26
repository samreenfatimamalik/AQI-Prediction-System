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

# ==================== DESIGN TOKENS ====================
# "Instrument panel" aesthetic — the dashboard reads like a sensor
# station readout rather than a generic SaaS dark theme.
VOID = "#0d0f11"          # app background — soot black
PANEL = "#17191c"         # card background
PANEL_RAISED = "#1c1f23"  # slightly raised surface
HAIRLINE = "#2a2d31"      # thin borders, instrument-panel style
TEXT_PRIMARY = "#e8e6e1"  # warm off-white, like dust/paper
TEXT_MUTED = "#8b8d90"
TEXT_FAINT = "#5a5d61"
ACCENT_HAZE = "#c9a15a"   # dusty gold — sun filtered through smog




CITIES = ["Lahore", "Karachi", "Islamabad", "Faisalabad", "Peshawar"]


def get_aqi_category(value):
    if value <= 50:
        return "Good", "#7fb069"
    elif value <= 100:
        return "Moderate", "#d4a54c"
    elif value <= 150:
        return "Unhealthy for Sensitive Groups", "#c97a3d"
    elif value <= 200:
        return "Unhealthy", "#b8503f"
    elif value <= 300:
        return "Very Unhealthy", "#8b5a8f"
    else:
        return "Hazardous", "#6b3838"


def grain_params(value):
    """Ties visual particulate 'grain' density to the AQI reading itself —
    worse air quality renders as denser, darker grain in the strip."""
    v = max(0, min(value, 300))
    opacity = 0.12 + (v / 300) * 0.55
    size = 10 - (v / 300) * 6
    return round(opacity, 2), round(size, 1)


def themed_chart(fig, height=420):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=PANEL,
        plot_bgcolor=PANEL,
        font=dict(family="IBM Plex Mono, monospace", color=TEXT_MUTED, size=12),
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", y=1.12, font=dict(color=TEXT_MUTED)),
    )
    fig.update_xaxes(gridcolor=HAIRLINE, zerolinecolor=HAIRLINE)
    fig.update_yaxes(gridcolor=HAIRLINE, zerolinecolor=HAIRLINE)
    return fig


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

    today = pd.Timestamp.now(tz="UTC").normalize()
    df = df[df["date"] <= today]

    cutoff = df["date"].max() - pd.Timedelta(days=15)
    return df[df["date"] >= cutoff].reset_index(drop=True)


def build_engineered_features(df):
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
    st.markdown(f"""
    <p style="font-family:'IBM Plex Mono',monospace; font-size:20px; font-weight:600;
    color:{TEXT_PRIMARY}; margin-bottom:0;">🌫️ PEARLS AQI</p>
    <p style="font-family:'IBM Plex Mono',monospace; font-size:11px; letter-spacing:1px;
    color:{TEXT_FAINT}; text-transform:uppercase; margin-top:2px;">Station Console</p>
    """, unsafe_allow_html=True)
    city = st.selectbox("Select a city", CITIES)
    if st.button("↻ Refresh feed", width='stretch'):
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
        st.caption(f"DATA AS OF {last_date.strftime('%Y-%m-%d')}")
        st.metric("Temperature", f"{latest_row['temperature'].iloc[0]:.1f}°C")
        st.metric("Humidity", f"{latest_row['humidity'].iloc[0]:.0f}%")
        st.metric("Wind Speed", f"{latest_row['wind_speed'].iloc[0]:.1f} km/h")
        st.metric("Pressure", f"{latest_row['pressure'].iloc[0]:.0f} hPa")

    # ---------------- HEADER ----------------
    st.markdown(f"""
    <p style="font-family:'IBM Plex Mono', monospace; font-size:12px; letter-spacing:2px;
    text-transform:uppercase; color:{TEXT_FAINT}; margin-bottom:2px;">Pearls AQI Predictor</p>
    """, unsafe_allow_html=True)
    st.title(f"{city}, Pakistan")
    st.caption("3-day particulate forecast · serverless ML pipeline")

    # ---------------- SENSOR READOUT (today's AQI) ----------------
    hist = city_df.sort_values("date")
    today_val = hist["pm2_5"].iloc[-1]
    today_cat, today_color = get_aqi_category(today_val)
    g_opacity, g_size = grain_params(today_val)

    st.markdown(f"""
    <div class="readout">
        <div>
            <p class="readout-eyebrow">Reading · {last_date.strftime('%A, %b %d')}</p>
            <h1 class="readout-value" style="color:{today_color}">{today_val:.0f}</h1>
            <p class="readout-status" style="color:{today_color}">{today_cat}</p>
        </div>
        <div class="readout-side">
            <p class="city">📍 {city}</p>
            <p class="cond">TEMP {latest_row['temperature'].iloc[0]:.1f}°C &nbsp;·&nbsp; RH {latest_row['humidity'].iloc[0]:.0f}%</p>
            <p class="cond">WIND {latest_row['wind_speed'].iloc[0]:.1f} km/h</p>
        </div>
    </div>
    <div class="grain-strip" style="color:{today_color}; opacity:{g_opacity}; background-size:{g_size}px {g_size}px;"></div>
    <div style="margin-bottom:28px;"></div>
    """, unsafe_allow_html=True)

    # ---------------- 3-DAY FORECAST ----------------
    st.markdown('<p class="sensor-label">3-Day Forecast Channels</p>', unsafe_allow_html=True)

    cols = st.columns(3)
    label_dates = {
        "1d": last_date + pd.Timedelta(days=1),
        "2d": last_date + pd.Timedelta(days=2),
        "3d": last_date + pd.Timedelta(days=3),
    }
    labels = {h: d.strftime("%a, %b %d") for h, d in label_dates.items()}
    hazard = False
    for i, h in enumerate(["1d", "2d", "3d"]):
        val = predictions[h]
        cat, color = get_aqi_category(val)
        if val > 150:
            hazard = True
        with cols[i]:
            st.markdown(f"""
            <div class="channel" style="border-left-color:{color};">
                <p class="date">{labels[h]}</p>
                <h1 class="val" style="color:{color}">{val:.0f}</h1>
                <p class="cat" style="color:{color}">{cat}</p>
            </div>
            """, unsafe_allow_html=True)

    st.write("")
    if hazard:
        st.markdown(f"""
        <div class="alert-banner" style="border-color:#b8503f; background-color:#b8503f1a; color:#e0a596;">
            ⚠ HAZARDOUS LEVELS PROJECTED — sensitive groups should limit outdoor exposure in {city}.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="alert-banner" style="border-color:#7fb069; background-color:#7fb0691a; color:#a8c99a;">
            ✓ NO HAZARDOUS LEVELS PROJECTED for {city} over the next 3 days.
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    tab1, tab2, tab3 = st.tabs(["Trend History", "Compare Cities", "Why This Prediction?"])

    # ---------------- TAB 1: TREND ----------------
    with tab1:
        forecast_dates = [last_date + pd.Timedelta(days=d) for d in [1, 2, 3]]
        forecast_vals = [predictions["1d"], predictions["2d"], predictions["3d"]]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hist["date"], y=hist["pm2_5"], mode="lines+markers",
            name="Observed PM2.5", line=dict(color=ACCENT_HAZE, width=2),
            marker=dict(size=5)
        ))
        fig.add_trace(go.Scatter(
            x=forecast_dates, y=forecast_vals, mode="lines+markers",
            name="Forecast", line=dict(color="#b8503f", dash="dash", width=2),
            marker=dict(size=7, symbol="diamond")
        ))
        fig = themed_chart(fig)
        fig.update_layout(xaxis_title="Date", yaxis_title="PM2.5 / AQI")
        st.plotly_chart(fig, width='stretch')

    # ---------------- TAB 2: COMPARE CITIES ----------------
    with tab2:
        comp_df = predict_all_cities(all_data, models)
        st.dataframe(comp_df, width='stretch', hide_index=True)

        fig2 = px.bar(
            comp_df.melt(id_vars=["City", "Status"], value_vars=["Tomorrow", "In 2 days", "In 3 days"],
                          var_name="Horizon", value_name="AQI"),
            x="City", y="AQI", color="Horizon", barmode="group",
            color_discrete_sequence=[ACCENT_HAZE, "#c97a3d", "#b8503f"],
        )
        fig2 = themed_chart(fig2)
        st.plotly_chart(fig2, width='stretch')
        
except Exception as e:
    st.error("An error occurred while loading the dashboard.")
    st.exception(e)