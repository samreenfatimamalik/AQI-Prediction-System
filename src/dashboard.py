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

# ==================== BASE CSS ====================
st.markdown(f"""
<style>
    .readout {{
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        background: {PANEL};
        border: 1px solid {HAIRLINE};
        border-radius: 10px;
        padding: 24px 28px;
    }}
    .readout-eyebrow {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 11px;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: {TEXT_FAINT};
        margin-bottom: 4px;
    }}
    .readout-value {{
        font-size: 64px;
        font-weight: 700;
        margin: 0;
        line-height: 1;
        font-family: 'IBM Plex Mono', monospace;
    }}
    .readout-status {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 14px;
        letter-spacing: 0.5px;
        margin-top: 4px;
    }}
    .readout-side {{
        text-align: right;
    }}
    .readout-side .city {{
        color: {TEXT_PRIMARY};
        font-size: 14px;
        margin: 0 0 6px 0;
    }}
    .readout-side .cond {{
        color: {TEXT_MUTED};
        font-family: 'IBM Plex Mono', monospace;
        font-size: 12px;
        margin: 2px 0;
    }}
    .grain-strip {{
        height: 6px;
        border-radius: 3px;
        background-image: radial-gradient(currentColor 1px, transparent 1px);
        margin-top: 10px;
    }}
    .sensor-label {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 11px;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: {TEXT_FAINT};
        margin-bottom: 10px;
    }}
    .channel {{
        background: {PANEL};
        border: 1px solid {HAIRLINE};
        border-left: 3px solid;
        border-radius: 8px;
        padding: 14px 16px;
    }}
    .channel .date {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 11px;
        color: {TEXT_FAINT};
        text-transform: uppercase;
        letter-spacing: 1px;
        margin: 0 0 6px 0;
    }}
    .channel .val {{
        font-size: 34px;
        font-weight: 700;
        margin: 0;
        font-family: 'IBM Plex Mono', monospace;
    }}
    .channel .cat {{
        font-size: 12px;
        margin: 4px 0 0 0;
        font-family: 'IBM Plex Mono', monospace;
    }}
    .alert-banner {{
        border: 1px solid;
        border-radius: 8px;
        padding: 12px 16px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 13px;
        letter-spacing: 0.3px;
    }}
    .pollutant-card {{
        background: {PANEL};
        border: 1px solid {HAIRLINE};
        border-radius: 8px;
        padding: 14px 16px;
        text-align: left;
    }}
    .pollutant-card .label {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 10px;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        color: {TEXT_FAINT};
        margin: 0 0 6px 0;
    }}
    .pollutant-card .value {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 22px;
        font-weight: 700;
        color: {TEXT_PRIMARY};
        margin: 0;
    }}
    .pollutant-card .unit {{
        font-size: 12px;
        color: {TEXT_MUTED};
        font-weight: 400;
        margin-left: 3px;
    }}
</style>
""", unsafe_allow_html=True)


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


def make_aqi_gauge(value, color):
    """Circular AQI gauge — a dial-style readout in the spirit of the
    reference design, color-banded to match AQI severity zones."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={
            "font": {"family": "IBM Plex Mono, monospace", "size": 46, "color": color},
            "suffix": ""
        },
        gauge={
            "axis": {
                "range": [0, 300],
                "tickwidth": 1,
                "tickcolor": HAIRLINE,
                "tickfont": {"family": "IBM Plex Mono, monospace", "size": 10, "color": TEXT_FAINT},
            },
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": PANEL,
            "borderwidth": 0,
            "steps": [
                {"range": [0, 50], "color": "#7fb06933"},
                {"range": [50, 100], "color": "#d4a54c33"},
                {"range": [100, 150], "color": "#c97a3d33"},
                {"range": [150, 200], "color": "#b8503f33"},
                {"range": [200, 300], "color": "#6b383833"},
            ],
        },
    ))
    fig.update_layout(
        height=220,
        margin=dict(l=20, r=20, t=20, b=10),
        paper_bgcolor=PANEL,
        font={"color": TEXT_MUTED, "family": "IBM Plex Mono, monospace"},
    )
    return fig


def pollutant_card(label, value, unit):
    st.markdown(f"""
    <div class="pollutant-card">
        <p class="label">{label}</p>
        <p class="value">{value}<span class="unit">{unit}</span></p>
    </div>
    """, unsafe_allow_html=True)


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


@st.cache_data(ttl=1800, show_spinner="Fetching latest data from Hopsworks...")
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

    readout_col, gauge_col = st.columns([1.3, 1])

    with readout_col:
        st.markdown(f"""
        <div class="readout" style="height: 100%;">
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
        """, unsafe_allow_html=True)

    with gauge_col:
        st.plotly_chart(make_aqi_gauge(today_val, today_color), width='stretch', config={"displayModeBar": False})

    st.write("")

    # ---------------- POLLUTANT GRID ----------------
    st.markdown('<p class="sensor-label">Sensor Readings</p>', unsafe_allow_html=True)
    p1, p2, p3, p4, p5, p6 = st.columns(6)
    with p1:
        pollutant_card("PM2.5", f"{hist['pm2_5'].iloc[-1]:.0f}", "µg/m³")
    with p2:
        pollutant_card("PM10", f"{hist['pm10'].iloc[-1]:.0f}", "µg/m³")
    with p3:
        pollutant_card("Temp", f"{latest_row['temperature'].iloc[0]:.1f}", "°C")
    with p4:
        pollutant_card("Humidity", f"{latest_row['humidity'].iloc[0]:.0f}", "%")
    with p5:
        pollutant_card("Wind", f"{latest_row['wind_speed'].iloc[0]:.1f}", "km/h")
    with p6:
        pollutant_card("Pressure", f"{latest_row['pressure'].iloc[0]:.0f}", "hPa")

    st.write("")

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

    # ---------------- HELPER: SHAP BACKGROUND DATA ----------------
    def get_background_data(all_data, feature_cols, n_samples=50):
        """Sample of recent rows (all cities) to give SHAP a baseline
        'what's typical' reference to compare each prediction against."""
        eng = build_engineered_features(all_data)
        eng = eng.dropna(subset=feature_cols)
        if len(eng) > n_samples:
            eng = eng.sample(n_samples, random_state=42)
        return eng[feature_cols]

    # ---------------- TAB 3: WHY THIS PREDICTION ----------------
    with tab3:
        st.markdown('<p class="sensor-label">SHAP Feature Contributions</p>', unsafe_allow_html=True)

        horizon_choice = st.radio(
            "Forecast horizon", ["1d", "2d", "3d"], horizontal=True,
            format_func=lambda h: {"1d": "Tomorrow", "2d": "In 2 days", "3d": "In 3 days"}[h]
        )

        shap_model = models[horizon_choice]
        feature_cols = list(shap_model.feature_names_in_)
        row_for_shap = latest_row[feature_cols]

        with st.spinner("Calculating feature contributions..."):
            background = get_background_data(all_data, feature_cols)
            explainer = get_shap_explainer(horizon_choice, shap_model, background)
            shap_values = explainer(row_for_shap)

        contributions = pd.Series(
            shap_values.values[0], index=feature_cols
        ).sort_values(key=abs, ascending=False)

        top_increase = contributions[contributions > 0].idxmax() if (contributions > 0).any() else None
        top_decrease = contributions[contributions < 0].idxmin() if (contributions < 0).any() else None

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Predicted AQI", f"{predictions[horizon_choice]:.1f}")
        with c2:
            if top_increase is not None:
                st.metric("Top increase", top_increase, f"+{contributions[top_increase]:.2f}")
        with c3:
            if top_decrease is not None:
                st.metric("Top decrease", top_decrease, f"{contributions[top_decrease]:.2f}")

        st.write("")

        # Horizontal bar of the top 8 contributing features (waterfall-style)
        N = 8
        top_n = contributions.head(N).sort_values()
        colors = ["#b8503f" if v > 0 else "#7fb069" for v in top_n.values]

        fig3 = go.Figure(go.Bar(
            x=top_n.values, y=top_n.index, orientation="h",
            marker_color=colors,
            text=[f"{v:+.2f}" for v in top_n.values], textposition="outside"
        ))
        fig3 = themed_chart(fig3, height=350)
        fig3.update_layout(xaxis_title="SHAP contribution to AQI prediction", yaxis_title="")
        st.plotly_chart(fig3, width='stretch')

        with st.expander(f"Show all {len(contributions)} features"):
            full_sorted = contributions.sort_values()
            colors_full = ["#b8503f" if v > 0 else "#7fb069" for v in full_sorted.values]
            fig_full = go.Figure(go.Bar(
                x=full_sorted.values, y=full_sorted.index, orientation="h",
                marker_color=colors_full,
            ))
            fig_full = themed_chart(fig_full, height=max(400, len(full_sorted) * 22))
            st.plotly_chart(fig_full, width='stretch')

        st.caption("Red bars push the prediction up · Green bars pull it down")

except Exception as e:
    st.error("An error occurred while loading the dashboard.")
    st.exception(e)