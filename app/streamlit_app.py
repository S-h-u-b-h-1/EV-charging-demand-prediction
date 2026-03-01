import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np

st.set_page_config(
    page_title="IntelliCharge360",
    layout="wide"
)

# -----------------------------
# HEADER
# -----------------------------
st.title("Intelligent EV Charging Demand & Infrastructure Planning")

st.markdown("""
A decision-support dashboard for EV infrastructure planners.  
Upload charging data to forecast demand, detect congestion risk, and generate infrastructure recommendations.
""")

st.divider()

# -----------------------------
# MODEL LOAD
# -----------------------------
MODEL_PATH = "models/best_ev_demand_model.pkl"

if not os.path.exists(MODEL_PATH):
    st.error("Model not found. Deployment configuration issue.")
    st.stop()

model = joblib.load(MODEL_PATH)

# -----------------------------
# FILE UPLOAD
# -----------------------------
st.subheader("Upload Charging Dataset")

uploaded_file = st.file_uploader(
    "Upload Processed Hourly EV Data (CSV)",
    type=["csv"]
)

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    required_features = [
        "station_encoded",
        "hour","dayofweek","month","day","weekofyear",
        "hour_sin","hour_cos","dow_sin","dow_cos",
        "lag_1","rolling_3h","rolling_24h"
    ]

    if not all(col in df.columns for col in required_features):
        st.error("Dataset missing required features.")
        st.stop()

    st.success("Data successfully loaded.")

    # -----------------------------
    # STATION SELECTION
    # -----------------------------
    station_list = sorted(df["station_encoded"].unique())

    selected_station = st.selectbox(
        "Select Station for Analysis",
        station_list
    )

    station_df = df[df["station_encoded"] == selected_station]

    # -----------------------------
    # PREDICTION
    # -----------------------------
    predictions = model.predict(station_df[required_features])
    station_df["Predicted_kWh"] = predictions

    avg_demand = station_df["Predicted_kWh"].mean()
    peak_demand = station_df["Predicted_kWh"].max()
    peak_hour = station_df.loc[
        station_df["Predicted_kWh"].idxmax(), "hour"
    ]

    # -----------------------------
    # EXECUTIVE DASHBOARD
    # -----------------------------
    st.subheader("Executive Summary")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Average Demand (kWh)", round(avg_demand,2))
    col2.metric("Peak Demand (kWh)", round(peak_demand,2))
    col3.metric("Peak Hour", f"{int(peak_hour)}:00")

    # Congestion Risk Logic
    if peak_demand > 25:
        risk = "High"
        risk_color = "🔴"
    elif peak_demand > 15:
        risk = "Moderate"
        risk_color = "🟠"
    else:
        risk = "Low"
        risk_color = "🟢"

    col4.metric("Congestion Risk", f"{risk_color} {risk}")

    st.divider()

    # -----------------------------
    # DEMAND VISUALIZATION
    # -----------------------------
    st.subheader("Hourly Demand Forecast")

    hourly_summary = (
        station_df.groupby("hour")["Predicted_kWh"]
        .mean()
        .reset_index()
    )

    st.line_chart(hourly_summary.set_index("hour"))

    # -----------------------------
    # INFRASTRUCTURE RECOMMENDATION
    # -----------------------------
    st.subheader("Infrastructure Recommendation")

    chargers_needed = int(np.ceil(peak_demand / 10))

    if risk == "High":
        recommendation = f"""
        This station shows high congestion risk.  
        Recommended additional chargers: **{chargers_needed} units**.  
        Consider load balancing or time-based pricing.
        """
    elif risk == "Moderate":
        recommendation = f"""
        Demand is moderate.  
        Monitor peak hours and prepare capacity expansion plan.
        """
    else:
        recommendation = """
        Current infrastructure is sufficient.  
        No immediate expansion required.
        """

    st.info(recommendation)

    # -----------------------------
    # BUSINESS IMPACT
    # -----------------------------
    st.subheader("Operational Insight")

    estimated_daily_energy = station_df["Predicted_kWh"].sum()
    estimated_revenue = estimated_daily_energy * 15  # assume ₹15 per kWh

    st.markdown(f"""
    • Estimated Daily Energy Delivered: **{round(estimated_daily_energy,2)} kWh**  
    • Estimated Daily Revenue (₹15/kWh): **₹{round(estimated_revenue,2)}**  
    """)

    # -----------------------------
    # DOWNLOAD REPORT
    # -----------------------------
    st.subheader("Download Planning Summary")

    summary_text = f"""
    Station: {selected_station}
    Average Demand: {round(avg_demand,2)} kWh
    Peak Demand: {round(peak_demand,2)} kWh
    Peak Hour: {peak_hour}:00
    Congestion Risk: {risk}
    Estimated Daily Revenue: ₹{round(estimated_revenue,2)}
    """

    st.download_button(
        label="Download Summary Report",
        data=summary_text,
        file_name="station_planning_summary.txt"
    )

else:
    st.info("Upload dataset to start intelligent demand analysis.")