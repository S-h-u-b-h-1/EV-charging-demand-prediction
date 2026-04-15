import sys
import os

# -----------------------------
# FIX IMPORT PATH
# -----------------------------
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

# -----------------------------
# IMPORTS
# -----------------------------
import streamlit as st
import pandas as pd
import joblib
import numpy as np

from agent.graph import run_agent

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="IntelliCharge360", layout="wide")

st.title("⚡ Intelligent EV Charging Demand & Infrastructure Planning")

st.markdown("""
Upload charging data → Predict demand → Detect congestion →  
🤖 AI Agent generates infrastructure + scheduling plan
""")

st.divider()

# -----------------------------
# LOAD MODEL
# -----------------------------
MODEL_PATH = "models/best_ev_demand_model.pkl"

if not os.path.exists(MODEL_PATH):
    st.error("Model not found.")
    st.stop()

model = joblib.load(MODEL_PATH)

# -----------------------------
# FILE UPLOAD
# -----------------------------
uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully.")

    # -----------------------------
    # FEATURE CHECK
    # -----------------------------
    required_features = [
        "station_encoded",
        "hour","dayofweek","month","day","weekofyear",
        "hour_sin","hour_cos","dow_sin","dow_cos",
        "lag_1","rolling_3h","rolling_24h"
    ]

    if not all(col in df.columns for col in required_features):
        st.error("Dataset missing required features.")
        st.stop()

    # -----------------------------
    # STATION SELECT
    # -----------------------------
    station_list = sorted(df["station_encoded"].unique())
    selected_station = st.selectbox("Select Station", station_list)

    station_df = df[df["station_encoded"] == selected_station].copy()

    # -----------------------------
    # PREDICTION
    # -----------------------------
    predictions = model.predict(station_df[required_features])
    station_df["Predicted_kWh"] = predictions

    avg_demand = station_df["Predicted_kWh"].mean()
    peak_demand = station_df["Predicted_kWh"].max()
    peak_hour = int(station_df.loc[
        station_df["Predicted_kWh"].idxmax(), "hour"
    ])

    # -----------------------------
    # RISK LOGIC
    # -----------------------------
    if peak_demand > 25:
        risk = "High"
        color = "🔴"
    elif peak_demand > 15:
        risk = "Moderate"
        color = "🟠"
    else:
        risk = "Low"
        color = "🟢"

    # -----------------------------
    # DASHBOARD
    # -----------------------------
    st.subheader("📊 Executive Summary")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg Demand", round(avg_demand,2))
    c2.metric("Peak Demand", round(peak_demand,2))
    c3.metric("Peak Hour", f"{peak_hour}:00")
    c4.metric("Risk", f"{color} {risk}")

    # -----------------------------
    # CHART
    # -----------------------------
    st.subheader("📈 Demand Forecast")

    hourly_summary = (
        station_df.groupby("hour")["Predicted_kWh"]
        .mean()
        .reset_index()
    )

    st.line_chart(hourly_summary.set_index("hour"))

    # -----------------------------
    # BASIC RECOMMENDATION
    # -----------------------------
    st.subheader("📌 Basic Recommendation")

    chargers_needed = int(np.ceil(peak_demand / 10))

    if risk == "High":
        st.warning(f"Add {chargers_needed} chargers + load balancing")
    elif risk == "Moderate":
        st.info("Monitor demand and plan expansion")
    else:
        st.success("Infrastructure sufficient")

    # -----------------------------
    # AGENT INPUT
    # -----------------------------
    demand_dict = {
        f"Hour_{int(h)}": float(v)
        for h, v in zip(station_df["hour"], station_df["Predicted_kWh"])
    }

    # -----------------------------
    # AGENT SECTION
    # -----------------------------
    st.divider()
    st.header("🤖 Agentic Infrastructure Planning")

    if st.button("Run AI Agent"):

        with st.spinner("Agent reasoning in progress..."):

            input_state = {
                "demand_forecast": demand_dict
            }

            result = run_agent(input_state)

        # -----------------------------
        # OUTPUT DISPLAY
        # -----------------------------
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🔥 High Load Zones")
            st.success(result.get("high_load_zones", []))

            st.subheader("🧠 Insights")
            for i in result.get("insights", []):
                st.write("•", i)

        with col2:
            st.subheader("🏗 Recommendations")
            for r in result.get("recommendations", []):
                st.info(r)

            st.subheader("⚡ Scheduling Plan")
            for s in result.get("scheduling_plan", []):
                st.write("→", s)

        # -----------------------------
        # STRUCTURED OUTPUT (VERY IMPORTANT)
        # -----------------------------
        st.subheader("📦 Structured JSON Output")
        st.json(result)

        # -----------------------------
        # RAG DEBUG (BONUS MARKS)
        # -----------------------------
        if "retrieved_docs" in result:
            st.subheader("📚 Retrieved Guidelines (RAG)")
            for doc in result["retrieved_docs"]:
                st.write("•", doc)

else:
    st.info("Upload dataset to begin.")