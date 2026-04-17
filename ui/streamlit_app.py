import os
import sys
import warnings

CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

import joblib
import pandas as pd
import streamlit as st

from agent.graph import run_agent_workflow
from ml.config import MODEL_PATH
from ml.evaluation import load_model_metrics
from utils.contracts import ARCHITECTURE_TEXT, INPUT_SPEC, OUTPUT_SPEC, WORKFLOW_TEXT
from utils.logger import get_logger
from utils.validation import validate_uploaded_dataframe

logger = get_logger(__name__)


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        logger.error("Model file not found at %s", MODEL_PATH)
        return None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return joblib.load(MODEL_PATH)
    except Exception as exc:
        logger.exception("Unable to load model: %s", exc)
        return None


def render_header():
    st.set_page_config(page_title="IntelliCharge360", layout="wide")
    st.title("Intelligent EV Charging Demand & Agentic Infrastructure Planning")
    st.caption("This system uses LangGraph-based stateful agent workflow.")
    st.markdown(
        """
Upload charging data -> Predict demand -> Detect hotspots -> Retrieve planning guidance ->
Reason over optimization trade-offs -> Generate an explainable infrastructure plan
"""
    )
    st.divider()


def render_architecture_and_specs():
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("System Architecture")
        st.code(ARCHITECTURE_TEXT, language="text")
        st.caption("Execution Workflow")
        st.code(WORKFLOW_TEXT, language="text")

    with col2:
        st.subheader("Input Specification")
        for item in INPUT_SPEC:
            st.write("•", item)

        st.subheader("Output Specification")
        for item in OUTPUT_SPEC:
            st.write("•", item)

    st.divider()


def render_model_metrics(metrics: dict[str, object]):
    st.subheader("Model Performance")
    if not metrics.get("available"):
        st.warning(metrics.get("interpretation", "Metrics unavailable."))
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", f"{metrics['mae']:.2f}")
    c2.metric("RMSE", f"{metrics['rmse']:.2f}")
    c3.metric("R² Score", f"{metrics['r2']:.3f}")
    st.info(metrics["interpretation"])
    st.caption(
        f"Evaluated on {metrics['rows_evaluated']} held-out rows using chronological split from {metrics['split_date']}."
    )
    st.divider()


def render_dashboard(result: dict):
    summary = result.get("summary", {})

    for error_message in result.get("errors", []):
        st.error(error_message)

    for warning_message in result.get("warnings", []):
        st.warning(warning_message)

    st.subheader("Executive Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg Demand", _fmt_metric(summary.get("avg_demand")))
    c2.metric("Peak Demand", _fmt_metric(summary.get("peak_demand")))
    c3.metric("Peak Hour", summary.get("peak_hour", "N/A"))
    c4.metric("Risk", summary.get("risk_level", "Unknown"))

    st.subheader("Demand Forecast")
    prediction_rows = result.get("predictions", [])
    if prediction_rows:
        forecast_df = pd.DataFrame(prediction_rows)
        hourly_summary = forecast_df.groupby("hour")["predicted_demand"].mean().reset_index()
        st.line_chart(hourly_summary.set_index("hour"))
    else:
        st.warning("Prediction output is unavailable.")

    left, right = st.columns(2)
    with left:
        st.subheader("Hotspot Detection")
        hotspots = result.get("hotspots", [])
        if hotspots:
            st.dataframe(pd.DataFrame(hotspots), use_container_width=True)
        else:
            st.warning("No congestion hotspots were identified.")

        st.subheader("Reasoning")
        for item in result.get("reasoning", []):
            st.write("•", item)

    with right:
        final_plan = result.get("final_plan", {})
        infrastructure_plan = final_plan.get("infrastructure_plan", [])
        schedule = final_plan.get("schedule", [])
        recommendations = final_plan.get("recommendations", [])

        st.subheader("Infrastructure Plan")
        if infrastructure_plan:
            st.dataframe(pd.DataFrame(infrastructure_plan), use_container_width=True)
        else:
            st.warning("No infrastructure plan generated.")

        st.subheader("Scheduling Strategy")
        if schedule:
            for item in schedule:
                st.write("→", item)
        else:
            st.warning("No scheduling plan generated.")

        st.subheader("Recommendations")
        if recommendations:
            for item in recommendations:
                st.success(item)
        else:
            st.warning("No recommendations generated.")

    st.subheader("Planning Explanation")
    explanation = final_plan.get("explanation") or result.get("reasoning_summary")
    if result.get("insufficient_data"):
        st.warning(explanation)
    else:
        st.info(explanation)

    st.subheader("Retrieved Guidance")
    for doc in result.get("retrieved_docs", []):
        if result.get("rag_fallback_used"):
            st.warning(doc)
        else:
            st.write("•", doc)

    with st.expander("Structured Output"):
        st.json(result)


def _fmt_metric(value):
    if value is None:
        return "N/A"
    return f"{value:.2f}"


def main():
    render_header()
    render_architecture_and_specs()
    render_model_metrics(load_model_metrics())

    model = load_model()
    if model is None:
        st.error("Model could not be loaded. Please verify the deployment artifacts.")
        return

    uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])
    if not uploaded_file:
        st.info("Upload dataset to begin.")
        return

    try:
        uploaded_df = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully.")
    except Exception as exc:
        logger.exception("Failed to read uploaded CSV: %s", exc)
        st.error("Unable to read the uploaded CSV file. Please upload a valid CSV dataset.")
        return

    is_valid, missing_optional, validation_error = validate_uploaded_dataframe(uploaded_df)
    if not is_valid:
        st.error(validation_error)
        return

    uploaded_df = uploaded_df.copy()
    uploaded_df["station_encoded"] = pd.to_numeric(uploaded_df["station_encoded"], errors="coerce")
    if uploaded_df["station_encoded"].isna().all():
        st.error("Uploaded dataset does not contain any valid numeric station identifiers.")
        return
    if uploaded_df["station_encoded"].isna().any():
        st.warning("Some station identifiers were invalid and have been ignored.")

    if missing_optional:
        st.warning(
            "Some model features are missing and will be imputed automatically: "
            + ", ".join(missing_optional)
        )

    station_list = sorted(uploaded_df["station_encoded"].dropna().astype(int).unique().tolist())
    if not station_list:
        st.error("Uploaded dataset does not contain any usable station identifiers.")
        return

    selected_station = st.selectbox("Select Station", station_list)

    if st.button("Run LangGraph Planning Agent"):
        with st.spinner("Stateful agent workflow in progress..."):
            result = run_agent_workflow(
                raw_data=uploaded_df.to_dict(orient="records"),
                selected_station=selected_station,
                model=model,
            )
        render_dashboard(result)


if __name__ == "__main__":
    main()
