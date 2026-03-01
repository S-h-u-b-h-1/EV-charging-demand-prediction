import sys, os
sys.path.append(os.path.abspath("."))

import streamlit as st
import numpy as np
import joblib

from src.train import train_pipeline


MODEL_PATH = "models/trained_model.pkl"
DATA_PATH = "data/raw/raw_dataset.csv"


st.title("EV Charging Demand Predictor ⚡")


# Train if model missing
if not os.path.exists(MODEL_PATH):

    st.write("Training model...")

    train_pipeline(DATA_PATH)

    st.success("Model trained successfully")


# Load model
model = joblib.load(MODEL_PATH)


# UI
station = st.number_input("Station", 0, 100, 1)
hour = st.slider("Hour", 0, 23, 12)
dow = st.slider("Day of week", 0, 6, 2)
month = st.slider("Month", 1, 12, 6)

lag1 = st.number_input("Lag 1", value=10.0)
roll3 = st.number_input("Rolling 3h", value=10.0)
roll24 = st.number_input("Rolling 24h", value=10.0)


hour_sin = np.sin(2*np.pi*hour/24)
hour_cos = np.cos(2*np.pi*hour/24)

dow_sin = np.sin(2*np.pi*dow/7)
dow_cos = np.cos(2*np.pi*dow/7)


features = np.array([[
    station,hour,dow,month,1,1,
    hour_sin,hour_cos,dow_sin,dow_cos,
    lag1,roll3,roll24
]])


if st.button("Predict"):

    pred = model.predict(features)[0]

    st.success(f"Prediction: {pred:.2f} kWh")