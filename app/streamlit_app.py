import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="EV Demand Prediction", layout="wide")

st.title("🚗 EV Charging Demand Prediction")

st.write("Upload model-ready CSV file to predict EV demand.")

# Load model (correct path)
model_path = os.path.join("models", "best_ev_demand_model.pkl")
model = joblib.load(model_path)

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data Preview")
    st.dataframe(df.head())

    features = [
        "station_encoded",
        "hour","dayofweek","month","day","weekofyear",
        "hour_sin","hour_cos","dow_sin","dow_cos",
        "lag_1","rolling_3h","rolling_24h"
    ]

    if all(col in df.columns for col in features):
        predictions = model.predict(df[features])
        df["Predicted_kWh"] = predictions

        st.subheader("Predictions")
        st.dataframe(df[["Predicted_kWh"]].head())

        st.line_chart(df["Predicted_kWh"])

    else:
        st.error("CSV does not contain required feature columns.")