import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
import numpy as np
import json

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Bharat Pulse 2.1",
    page_icon="🇮🇳",
    layout="wide"
)

st.markdown("""
<style>
.main { background-color: #f5f7f9; }
.stMetric {
    background-color: #ffffff;
    padding: 15px;
    border-radius: 10px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD STATE–DISTRICT MAP ----------------
@st.cache_data
def load_state_district_map():
    with open("state_district_map.json", "r") as f:
        return json.load(f)

STATE_DISTRICT_MAP = load_state_district_map()

# ---------------- LOAD MODEL & ENCODERS ----------------
@st.cache_resource
def load_assets():
    model = joblib.load("models/price_model.pkl")
    le_state = joblib.load("models/le_state.pkl")
    le_district = joblib.load("models/le_district.pkl")
    le_commodity = joblib.load("models/le_commodity.pkl")
    return model, le_state, le_district, le_commodity

try:
    model, le_state, le_district, le_commodity = load_assets()
except Exception:
    st.error("❌ Models not found. Ensure all .pkl files exist in /models.")
    st.stop()

# ---------------- UI HEADER ----------------
st.title("🇮🇳 Bharat Pulse 2.1")
st.markdown("### AI-based food price forecasting with uncertainty awareness")

col1, col2 = st.columns([1, 2])

# ---------------- INPUT PANEL ----------------
with col1:
    st.subheader("📍 Market Selection")

    state = st.selectbox(
        "Select State",
        sorted(STATE_DISTRICT_MAP.keys())
    )

    district = st.selectbox(
        "Select District",
        STATE_DISTRICT_MAP[state]
    )

    commodity = st.selectbox(
        "Select Commodity",
        sorted(le_commodity.classes_)
    )

    st.divider()

    st.subheader("💰 Market Intelligence")

    current_price = st.number_input(
        "Current Modal Price (per Quintal)",
        min_value=100,
        max_value=20000,
        value=2000
    )

    last_week_price = st.number_input(
        "Price 7 Days Ago",
        min_value=100,
        max_value=20000,
        value=1900
    )

# ---------------- OUTPUT PANEL ----------------
with col2:
    st.subheader("🔮 Forecast Results")
    st.caption(
        "⚠️ Forecast assumes stable market conditions based on recent trends. "
        "Sudden shocks (weather, supply, policy) may cause deviations."
    )

    if st.button("Generate 7-Day Forecast"):

        # -------- BASE INPUT --------
        base_input = pd.DataFrame([{
            "state_enc": le_state.transform([state])[0],
            "district_enc": le_district.transform([district])[0],
            "commodity_enc": le_commodity.transform([commodity])[0],
            "Modal_Price": current_price,
            "price_lag_7": last_week_price,
            "month": datetime.now().month
        }])

        # -------- POINT FORECAST --------
        point_prediction = model.predict(base_input)[0]
        change_pct = ((point_prediction - current_price) / current_price) * 100

        # -------- CONFIDENCE RANGE (RF ENSEMBLE) --------
        tree_preds = np.array([
            tree.predict(base_input)[0] for tree in model.estimators_
        ])
        lower_bound = np.percentile(tree_preds, 10)
        upper_bound = np.percentile(tree_preds, 90)

        # -------- MULTI-DAY FORECAST (RECURSIVE + CAP) --------
        DAILY_CAP = 0.06  # ±6%
        future_prices = []
        prev_price = current_price
        prev_lag = last_week_price

        for _ in range(7):
            step_input = pd.DataFrame([{
                "state_enc": le_state.transform([state])[0],
                "district_enc": le_district.transform([district])[0],
                "commodity_enc": le_commodity.transform([commodity])[0],
                "Modal_Price": prev_price,
                "price_lag_7": prev_lag,
                "month": datetime.now().month
            }])

            raw_pred = model.predict(step_input)[0]
            max_allowed = prev_price * (1 + DAILY_CAP)
            min_allowed = prev_price * (1 - DAILY_CAP)

            next_price = min(max(raw_pred, min_allowed), max_allowed)
            future_prices.append(next_price)

            prev_lag = prev_price
            prev_price = next_price

        # -------- METRICS --------
        m1, m2 = st.columns(2)
        m1.metric("Current Price", f"₹{current_price}")
        m2.metric(
            "7-Day Point Forecast",
            f"₹{int(point_prediction)}",
            f"{change_pct:.2f}%"
        )

        st.info(
            f"📊 Expected Price Range (7-Day): "
            f"₹{int(lower_bound)} – ₹{int(upper_bound)}"
        )

        # -------- CHART --------
        days = ["Today"] + [f"Day +{i}" for i in range(1, 8)]
        prices = [current_price] + future_prices

        chart_df = pd.DataFrame({
            "Timeline": days,
            "Price (INR)": prices
        })

        st.write("#### 📈 7-Day Price Trajectory")
        st.line_chart(chart_df.set_index("Timeline"))

# ---------------- FOOTER ----------------
st.divider()
st.caption(
    "Bharat Pulse 2.1 | Decision-support AI system | "
    "Random Forest Regressor"
)
