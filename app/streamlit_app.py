import sys
import os

# Fix import paths for Streamlit Cloud
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import streamlit as st
import numpy as np
import joblib

# Optional: only if you really want training capability
# from train_pipeline import train_pipeline


# Correct absolute paths
MODEL_PATH = os.path.join(BASE_DIR, "models", "trained_model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "raw_dataset.csv")


# Page config
st.set_page_config(page_title="EV Charging Demand Predictor", page_icon="⚡")


st.title("EV Charging Demand Predictor ⚡")


# Load model safely with caching
@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found at {MODEL_PATH}")
        st.stop()

    model = joblib.load(MODEL_PATH)
    return model


# Load model
model = load_model()

st.success("Model loaded successfully ✅")


# UI Inputs
st.header("Enter Features")

station = st.number_input("Station ID", min_value=0, max_value=100, value=1)

hour = st.slider("Hour", 0, 23, 12)

dow = st.slider("Day of week", 0, 6, 2)

month = st.slider("Month", 1, 12, 6)

lag1 = st.number_input("Lag 1 hour demand", value=10.0)

roll3 = st.number_input("Rolling 3-hour demand", value=10.0)

roll24 = st.number_input("Rolling 24-hour demand", value=10.0)


# Feature engineering
hour_sin = np.sin(2 * np.pi * hour / 24)
hour_cos = np.cos(2 * np.pi * hour / 24)

dow_sin = np.sin(2 * np.pi * dow / 7)
dow_cos = np.cos(2 * np.pi * dow / 7)


# Feature vector
features = np.array([[
    station,
    hour,
    dow,
    month,
    1,   # day placeholder
    1,   # week placeholder
    hour_sin,
    hour_cos,
    dow_sin,
    dow_cos,
    lag1,
    roll3,
    roll24
]])


# Prediction button
if st.button("Predict Demand"):

    try:

        prediction = model.predict(features)[0]

        st.success(f"Predicted Charging Demand: {prediction:.2f} kWh ⚡")

    except Exception as e:

        st.error(f"Prediction failed: {e}")
