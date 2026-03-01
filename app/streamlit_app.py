import sys
import os

# Get project root directory
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add root to python path
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import streamlit as st
import numpy as np
import joblib

# Correct absolute model path
MODEL_PATH = os.path.join(ROOT_DIR, "models", "trained_model.pkl")

st.title("EV Charging Demand Predictor ⚡")

# Debug info
st.write("Root directory:", ROOT_DIR)
st.write("Model path:", MODEL_PATH)

# Check model exists
if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file not found. Ensure trained_model.pkl is in models folder.")
    st.stop()

# Load model safely
try:
    model = joblib.load(MODEL_PATH)
    st.success("✅ Model loaded successfully")
except Exception as e:
    st.error(f"❌ Model loading failed: {e}")
    st.stop()


# UI inputs
station = st.number_input("Station ID", 0, 100, 1)
hour = st.slider("Hour", 0, 23, 12)
dow = st.slider("Day of Week", 0, 6, 2)
month = st.slider("Month", 1, 12, 6)

lag1 = st.number_input("Lag 1", value=10.0)
roll3 = st.number_input("Rolling 3h", value=10.0)
roll24 = st.number_input("Rolling 24h", value=10.0)

# Feature engineering
hour_sin = np.sin(2*np.pi*hour/24)
hour_cos = np.cos(2*np.pi*hour/24)

dow_sin = np.sin(2*np.pi*dow/7)
dow_cos = np.cos(2*np.pi*dow/7)

features = np.array([[
    station,
    hour,
    dow,
    month,
    1,
    1,
    hour_sin,
    hour_cos,
    dow_sin,
    dow_cos,
    lag1,
    roll3,
    roll24
]])

# Prediction
if st.button("Predict Demand"):
    try:
        prediction = model.predict(features)[0]
        st.success(f"Predicted demand: {prediction:.2f} kWh ⚡")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
