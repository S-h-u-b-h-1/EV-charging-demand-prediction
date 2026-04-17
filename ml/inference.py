from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ml.config import MODEL_FEATURES
from utils.logger import get_logger

logger = get_logger(__name__)

PREDICTION_FAILURE_MESSAGE = (
    "Prediction failed due to invalid data format or model issue. Please verify dataset."
)


def run_prediction(model: Any, feature_frame: pd.DataFrame) -> tuple[np.ndarray | None, str | None]:
    try:
        predictions = model.predict(feature_frame[MODEL_FEATURES])
        predictions = np.asarray(predictions, dtype=float)

        if predictions.ndim != 1 or len(predictions) == 0 or np.isnan(predictions).all():
            raise ValueError("Model returned empty or invalid predictions.")

        return np.clip(predictions, a_min=0.0, a_max=None), None
    except Exception as exc:
        logger.exception("Prediction failed: %s", exc)
        return None, PREDICTION_FAILURE_MESSAGE
