from __future__ import annotations

import os

MODEL_FEATURES = [
    "station_encoded",
    "hour",
    "dayofweek",
    "month",
    "day",
    "weekofyear",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "lag_1",
    "rolling_3h",
    "rolling_24h",
]

CORE_REQUIRED_COLUMNS = ["station_encoded", "hour"]

DEFAULT_FEATURE_VALUES = {
    "dayofweek": 0,
    "month": 1,
    "day": 1,
    "weekofyear": 1,
    "hour_sin": 0.0,
    "hour_cos": 1.0,
    "dow_sin": 0.0,
    "dow_cos": 1.0,
    "lag_1": 0.0,
    "rolling_3h": 0.0,
    "rolling_24h": 0.0,
}

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(ROOT_DIR, "models", "best_ev_demand_model.pkl")
PROCESSED_DATA_PATH = os.path.join(ROOT_DIR, "data", "processed", "model_ready_hourly_data.csv")
TARGET_COLUMN = "total_kWh"
