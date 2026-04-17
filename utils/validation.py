from __future__ import annotations

from typing import Any

import pandas as pd

from ml.config import CORE_REQUIRED_COLUMNS, DEFAULT_FEATURE_VALUES, MODEL_FEATURES
from utils.logger import get_logger

logger = get_logger(__name__)


def format_missing_columns_message(columns: list[str]) -> str:
    return (
        "Uploaded dataset is missing required columns: "
        f"{columns}. Please upload a valid EV charging dataset containing "
        "features like time, demand, station_id, and the model-ready inputs."
    )


def validate_uploaded_dataframe(df: pd.DataFrame) -> tuple[bool, list[str], str | None]:
    missing_required = [column for column in CORE_REQUIRED_COLUMNS if column not in df.columns]
    if missing_required:
        logger.warning("CSV validation failed. Missing required columns: %s", missing_required)
        return False, [], format_missing_columns_message(missing_required)

    missing_optional = [column for column in MODEL_FEATURES if column not in df.columns]
    if missing_optional:
        logger.warning("CSV uploaded with missing optional model features: %s", missing_optional)

    return True, missing_optional, None


def prepare_feature_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    prepared_df = df.copy()
    warnings: list[str] = []

    for column in MODEL_FEATURES:
        if column not in prepared_df.columns:
            default_value = DEFAULT_FEATURE_VALUES.get(column, 0.0)
            prepared_df[column] = default_value
            warnings.append(
                f"Missing feature `{column}` was filled with default value {default_value}."
            )

        prepared_df[column] = pd.to_numeric(prepared_df[column], errors="coerce")

        if prepared_df[column].isna().any():
            fill_value = _get_fill_value(prepared_df, column)
            prepared_df[column] = prepared_df[column].fillna(fill_value)
            warnings.append(
                f"Invalid or missing values in `{column}` were imputed with {fill_value}."
            )

    prepared_df["hour"] = prepared_df["hour"].clip(lower=0, upper=23).round().astype(int)
    prepared_df["station_encoded"] = prepared_df["station_encoded"].round().astype(int)

    return prepared_df, list(dict.fromkeys(warnings))


def build_data_quality_summary(df: pd.DataFrame, warnings: list[str]) -> dict[str, Any]:
    hours_covered = int(df["hour"].nunique()) if "hour" in df.columns else 0
    trend_columns = ["lag_1", "rolling_3h", "rolling_24h"]
    has_trend_signal = bool(
        set(trend_columns).issubset(df.columns)
        and df[trend_columns].notna().any().all()
    )

    insufficient_reasons = []
    if len(df) < 3:
        insufficient_reasons.append("Too few records are available for trend estimation.")
    if hours_covered < 2:
        insufficient_reasons.append("Hourly coverage is too narrow to distinguish peak and off-peak behavior.")
    if not has_trend_signal:
        insufficient_reasons.append("Demand trend features are missing or unusable.")

    return {
        "warnings": warnings,
        "hours_covered": hours_covered,
        "has_trend_signal": has_trend_signal,
        "insufficient_reasons": insufficient_reasons,
        "insufficient_data": bool(insufficient_reasons),
    }


def _get_fill_value(df: pd.DataFrame, column: str) -> float | int:
    if column == "hour":
        return 0
    if column == "station_encoded":
        modes = df[column].mode(dropna=True)
        return int(modes.iloc[0]) if not modes.empty else 0

    median_value = df[column].median()
    if pd.isna(median_value):
        return DEFAULT_FEATURE_VALUES.get(column, 0.0)
    return float(median_value)
