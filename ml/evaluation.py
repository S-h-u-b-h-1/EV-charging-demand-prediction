from __future__ import annotations

import warnings
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ml.config import MODEL_FEATURES, MODEL_PATH, PROCESSED_DATA_PATH, TARGET_COLUMN
from utils.logger import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def load_model_metrics() -> dict[str, object]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = joblib.load(MODEL_PATH)

        df = pd.read_csv(PROCESSED_DATA_PATH)
        df["hour_timestamp"] = pd.to_datetime(df["hour_timestamp"], errors="coerce")
        df = df.dropna(subset=["hour_timestamp"])

        split_index = int(len(df) * 0.8)
        split_date = df["hour_timestamp"].sort_values().iloc[split_index]
        test_df = df[df["hour_timestamp"] >= split_date].copy()

        predictions = model.predict(test_df[MODEL_FEATURES])
        mae = float(mean_absolute_error(test_df[TARGET_COLUMN], predictions))
        rmse = float(np.sqrt(mean_squared_error(test_df[TARGET_COLUMN], predictions)))
        r2 = float(r2_score(test_df[TARGET_COLUMN], predictions))

        interpretation = _build_interpretation(r2, rmse)

        return {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "rows_evaluated": int(len(test_df)),
            "split_date": str(split_date),
            "interpretation": interpretation,
            "available": True,
        }
    except Exception as exc:
        logger.exception("Failed to compute model evaluation metrics: %s", exc)
        return {
            "mae": None,
            "rmse": None,
            "r2": None,
            "rows_evaluated": 0,
            "split_date": None,
            "interpretation": "Model performance metrics are currently unavailable.",
            "available": False,
        }


def _build_interpretation(r2: float, rmse: float) -> str:
    explained_variance = max(0.0, min(1.0, r2)) * 100
    return (
        f"Model explains approximately {explained_variance:.1f}% of the variance in held-out hourly demand "
        f"with an RMSE of {rmse:.2f} kWh."
    )
