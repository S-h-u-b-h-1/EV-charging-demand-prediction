import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np

st.set_page_config(page_title="IntelliCharge360", layout="wide")

# ---------------------------
# HEADER
# ---------------------------
st.title("Intelligent EV Charging Demand Prediction & Infrastructure Planning")

st.markdown("""
This platform analyzes EV charging usage data and predicts future demand patterns.
It helps planners identify peak demand hours and make informed infrastructure decisions.
""")

st.divider()

# ---------------------------
# MODEL LOADING
# ---------------------------
MODEL_PATH = os.path.join("models", "best_ev_demand_model.pkl")

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found. Please ensure the model is correctly deployed.")
    st.stop()

model = joblib.load(MODEL_PATH)

# ---------------------------
# FILE UPLOAD
# ---------------------------
st.subheader("Upload Charging Data")

uploaded_file = st.file_uploader(
    "Upload model-ready CSV file",
    type=["csv"],
    help="Upload the processed hourly EV demand dataset."
)

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.success("File uploaded successfully.")

    # ---------------------------
    # REQUIRED FEATURES
    # ---------------------------
    features = [
        "station_encoded",
        "hour","dayofweek","month","day","weekofyear",
        "hour_sin","hour_cos","dow_sin","dow_cos",
        "lag_1","rolling_3h","rolling_24h"
    ]

    if not all(col in df.columns for col in features):
        st.error("Uploaded file does not contain required features.")
        st.stop()

    # ---------------------------
    # STATION SELECTION
    # ---------------------------
    station_list = df["station_encoded"].unique()

    selected_station = st.selectbox(
        "Select Charging Station",
        station_list
    )

    station_df = df[df["station_encoded"] == selected_station]

    # ---------------------------
    # PREDICTION
    # ---------------------------
    predictions = model.predict(station_df[features])
    station_df["Predicted_kWh"] = predictions

    # ---------------------------
    # DASHBOARD
    # ---------------------------
    st.subheader("Demand Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Average Predicted Demand (kWh)",
        round(station_df["Predicted_kWh"].mean(), 2)
    )

    col2.metric(
        "Peak Demand (kWh)",
        round(station_df["Predicted_kWh"].max(), 2)
    )

    peak_hour = station_df.loc[
        station_df["Predicted_kWh"].idxmax(), "hour"
    ]

    col3.metric(
        "Peak Hour of Day",
        f"{int(peak_hour)}:00"
    )

    # ---------------------------
    # DEMAND GRAPH
    # ---------------------------
    st.subheader("Hourly Demand Forecast")

    hourly_summary = (
        station_df.groupby("hour")["Predicted_kWh"]
        .mean()
        .reset_index()
    )

    st.line_chart(
        hourly_summary.set_index("hour")
    )

    # ---------------------------
    # INFRASTRUCTURE INSIGHT
    # ---------------------------
    st.subheader("Infrastructure Recommendation")

    avg_demand = station_df["Predicted_kWh"].mean()

    if avg_demand > 20:
        recommendation = "High demand station. Consider installing additional charging units."
    elif avg_demand > 10:
        recommendation = "Moderate demand. Monitor peak congestion before scaling."
    else:
        recommendation = "Low demand. Existing infrastructure sufficient."

    st.info(recommendation)

    # ---------------------------
    # RAW DATA EXPANDER
    # ---------------------------
    with st.expander("View Detailed Predictions"):
        st.dataframe(station_df.head(50))

else:
    st.info("Please upload a CSV file to begin analysis.")