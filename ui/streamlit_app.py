from __future__ import annotations

import json
import os
import sys
import warnings
from typing import Any

import joblib
import pandas as pd
import streamlit as st

CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

from agent.graph import run_agent_workflow
from ml.config import MODEL_PATH
from ml.evaluation import load_model_metrics
from utils.contracts import ARCHITECTURE_TEXT, INPUT_SPEC, OUTPUT_SPEC, WORKFLOW_TEXT
from utils.logger import get_logger
from utils.validation import validate_uploaded_dataframe

logger = get_logger(__name__)


st.set_page_config(
    page_title="IntelliCharge360",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


APP_CSS = """
<style>
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }
    .hero-card {
        border: 1px solid rgba(128,128,128,0.18);
        border-radius: 18px;
        padding: 1.2rem 1.25rem;
        background: linear-gradient(135deg, rgba(62,84,172,0.10), rgba(24,144,255,0.06));
        box-shadow: 0 8px 30px rgba(0,0,0,0.04);
    }
    .mini-card {
        border: 1px solid rgba(128,128,128,0.16);
        border-radius: 14px;
        padding: 0.85rem 1rem;
        background: rgba(255,255,255,0.55);
    }
    .muted-note {
        color: #667085;
        font-size: 0.92rem;
    }
    .section-gap {
        margin-top: 0.4rem;
        margin-bottom: 0.4rem;
    }
</style>
"""


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


@st.cache_resource
def load_model_features() -> list[str]:
    try:
        from ml.config import MODEL_FEATURES

        return list(MODEL_FEATURES)
    except Exception:
        return []


def _fmt_metric(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.2f}"
    except Exception:
        return "N/A"


def _download_json_button(payload: dict[str, Any], filename: str = "charge_sense_output.json") -> None:
    st.download_button(
        label="⬇️ Download structured output",
        data=json.dumps(payload, indent=2, ensure_ascii=False),
        file_name=filename,
        mime="application/json",
        use_container_width=True,
    )


def _sidebar_help() -> None:
    with st.sidebar:
        st.markdown("## ⚡ IntelliCharge360")
        st.caption("Agentic EV charging demand planning")
        st.markdown(
            """
**How to use**
1. Upload the CSV dataset
2. Select a station
3. Run the agent
4. Review forecast, hotspots, reasoning, and plan
"""
        )
        st.markdown("---")
        st.markdown("### What the app does")
        st.write("• Predicts EV charging demand")
        st.write("• Detects hotspots")
        st.write("• Retrieves planning guidelines")
        st.write("• Generates a charger + scheduling plan")
        st.write("• Uses Groq LLM for final reasoning when available")
        st.markdown("---")
        st.markdown("### Quick checks")
        st.write("• Python 3.10 recommended")
        st.write("• GROQ_API_KEY should be set in .env or Secrets")
        st.write("• Use the same uploaded file format as training")


def render_header() -> None:
    st.markdown(APP_CSS, unsafe_allow_html=True)

    st.markdown(
        """
<div class="hero-card">
    <h1 style="margin-bottom:0.25rem;">Intelligent EV Charging Demand & Agentic Infrastructure Planning</h1>
    <p style="margin-top:0; margin-bottom:0.5rem; font-size:1.02rem;">
        Upload charging data → predict demand → detect hotspots → retrieve planning guidance → reason with LLM + rules → generate a deployment plan.
    </p>
    <p class="muted-note" style="margin-bottom:0;">
        A Streamlit decision-support dashboard built for EV station planning, scheduling optimization, and explainable infrastructure recommendations.
    </p>
</div>
""",
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Workflow", "LangGraph")
    c2.metric("Retrieval", "FAISS RAG")
    c3.metric("Reasoning", "Rules + LLM")
    c4.metric("UI", "Streamlit")

    st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)


def render_overview(metrics: dict[str, Any]) -> None:
    st.markdown("## 🔎 Overview")

    left, right = st.columns([1.1, 0.9], gap="large")
    with left:
        st.markdown("### What this system does")
        st.write(
            "This system predicts EV charging demand, identifies high-load periods, retrieves planning guidance, and generates a structured infrastructure plan with scheduling recommendations."
        )
        st.write(
            "It combines classical ML for forecasting with an agentic workflow for explainable EV infrastructure planning."
        )

        st.markdown("### Why this project is strong")
        st.write("• Real-world sustainability and grid-planning use case")
        st.write("• Clear ML + agentic AI separation")
        st.write("• Retrieval-grounded reasoning")
        st.write("• Robust fallbacks for deployment stability")

    with right:
        st.markdown("### Model performance")
        if not metrics.get("available"):
            st.warning(metrics.get("interpretation", "Metrics unavailable."))
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric("MAE", f"{metrics['mae']:.2f}")
            m2.metric("RMSE", f"{metrics['rmse']:.2f}")
            m3.metric("R²", f"{metrics['r2']:.3f}")
            st.info(metrics["interpretation"])
            st.caption(
                f"Evaluated on {metrics['rows_evaluated']} held-out rows using chronological split from {metrics['split_date']}."
            )


def render_architecture_and_specs() -> None:
    st.markdown("## 🧩 System Architecture")

    tabs = st.tabs(["Architecture", "Workflow", "Input", "Output"])

    with tabs[0]:
        st.code(ARCHITECTURE_TEXT, language="text")

    with tabs[1]:
        st.code(WORKFLOW_TEXT, language="text")

    with tabs[2]:
        st.markdown("### Input specification")
        for item in INPUT_SPEC:
            st.write("•", item)

    with tabs[3]:
        st.markdown("### Output specification")
        for item in OUTPUT_SPEC:
            st.write("•", item)


def render_forecast_section(result: dict[str, Any]) -> None:
    st.markdown("## 📈 Forecast & Hotspots")

    col1, col2 = st.columns([1.4, 0.6], gap="large")
    with col1:
        prediction_rows = result.get("predictions", [])
        if prediction_rows:
            forecast_df = pd.DataFrame(prediction_rows)
            if "hour" in forecast_df.columns and "predicted_demand" in forecast_df.columns:
                hourly_summary = forecast_df.groupby("hour")["predicted_demand"].mean().reset_index()
                st.line_chart(hourly_summary.set_index("hour"))
            else:
                st.warning("Prediction output is missing the required columns for charting.")
        else:
            st.warning("Prediction output is unavailable.")

    with col2:
        summary = result.get("summary", {})
        st.markdown("### Executive summary")
        st.metric("Avg demand", _fmt_metric(summary.get("avg_demand")))
        st.metric("Peak demand", _fmt_metric(summary.get("peak_demand")))
        st.metric("Peak hour", summary.get("peak_hour", "N/A"))
        st.metric("Risk", summary.get("risk_level", "Unknown"))
        if summary.get("confidence"):
            st.caption(f"Confidence: {summary['confidence']}")

    hotspots = result.get("hotspots", [])
    if hotspots:
        st.markdown("### Hotspot detection")
        st.dataframe(pd.DataFrame(hotspots), use_container_width=True)
    else:
        st.info("No congestion hotspots were identified.")


def render_reasoning_section(result: dict[str, Any]) -> None:
    st.markdown("## 🧠 Reasoning & LLM Insight")

    llm_used = result.get("llm_used", False)
    if llm_used:
        st.success("AI reasoning enabled: Groq LLM was used in the planning step.")
    else:
        st.warning("LLM fallback used: the app relied on rule-based reasoning for this run.")

    reasoning = result.get("reasoning", [])
    for item in reasoning:
        if "LLM Insight" in item:
            st.markdown("### 🤖 AI Insight")
            st.info(item)
        else:
            st.write("•", item)

    if result.get("rag_reason"):
        st.caption(f"RAG note: {result.get('rag_reason')}")


def render_plan_section(result: dict[str, Any]) -> None:
    st.markdown("## 🏗️ Infrastructure Plan")

    final_plan = result.get("final_plan", {})
    infrastructure_plan = final_plan.get("infrastructure_plan", [])
    schedule = final_plan.get("schedule", [])
    recommendations = final_plan.get("recommendations", [])
    explanation = final_plan.get("explanation") or result.get("reasoning_summary")

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown("### Deployment plan")
        if infrastructure_plan:
            st.dataframe(pd.DataFrame(infrastructure_plan), use_container_width=True)
        else:
            st.warning("No infrastructure plan generated.")

        st.markdown("### Scheduling strategy")
        if schedule:
            for item in schedule:
                st.write("→", item)
        else:
            st.warning("No scheduling plan generated.")

    with right:
        st.markdown("### Recommendations")
        if recommendations:
            for item in recommendations:
                st.success(item)
        else:
            st.warning("No recommendations generated.")

        st.markdown("### Why this charger count?")
        if result.get("insufficient_data"):
            st.warning(explanation)
        else:
            st.info(explanation)

    if result.get("retrieved_docs"):
        st.markdown("### Retrieved guidance")
        for doc in result.get("retrieved_docs", []):
            if result.get("rag_fallback_used"):
                st.warning(doc)
            else:
                st.write("•", doc)


def render_debug_and_export(result: dict[str, Any]) -> None:
    st.markdown("## 🧾 Structured Output")
    c1, c2 = st.columns([0.7, 0.3], gap="large")
    with c1:
        st.json(result)
    with c2:
        _download_json_button(result)
        st.caption("Download the full structured response for report screenshots or debugging.")


def render_model_notes(model: Any) -> None:
    features = load_model_features()
    st.markdown("## 📊 Model Insight")
    st.write(
        "The selected model was chosen for a strong balance of accuracy, generalization, and deployment simplicity."
    )

    if features:
        st.caption(f"Model features: {', '.join(features)}")

    try:
        if hasattr(model, "feature_importances_"):
            st.markdown("### 🔍 Feature importance")
            feature_names = list(getattr(model, "feature_names_in_", []))
            if not feature_names:
                feature_names = [f"feature_{i}" for i in range(len(model.feature_importances_))]
            importance = pd.Series(model.feature_importances_, index=feature_names).sort_values()
            st.bar_chart(importance)
    except Exception as exc:
        logger.warning("Feature importance could not be rendered: %s", exc)
        st.caption("Feature importance is not available for this model.")


def render_input_panel(uploaded_df: pd.DataFrame) -> tuple[pd.DataFrame, int | None]:
    st.markdown("## 📥 Input")

    c1, c2 = st.columns([1.1, 0.9], gap="large")
    with c1:
        st.markdown("### Uploaded dataset preview")
        st.dataframe(uploaded_df.head(10), use_container_width=True)

    with c2:
        st.markdown("### Station selection")
        station_list = sorted(uploaded_df["station_encoded"].dropna().astype(int).unique().tolist())
        if not station_list:
            st.error("Uploaded dataset does not contain any usable station identifiers.")
            return uploaded_df, None

        selected_station = st.selectbox("Select Station", station_list)
        st.caption(f"Stations found: {len(station_list)}")

    return uploaded_df, selected_station


def main() -> None:
    _sidebar_help()
    render_header()
    render_architecture_and_specs()
    render_overview(load_model_metrics())

    model = load_model()
    if model is None:
        st.error("Model could not be loaded. Please verify the deployment artifacts.")
        return

    render_model_notes(model)

    st.markdown("## 🚀 Run the Agent")
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

    uploaded_df, selected_station = render_input_panel(uploaded_df)
    if selected_station is None:
        return

    st.markdown("### Run settings")
    c1, c2 = st.columns([0.7, 0.3])
    with c1:
        run_btn = st.button("▶ Run LangGraph Planning Agent", type="primary", use_container_width=True)
    with c2:
        st.checkbox("Show raw JSON after run", value=True, key="show_raw_json")

    if run_btn:
        status = st.status("Running demand prediction and planning workflow...", expanded=True)
        progress_bar = st.progress(0)
        try:
            status.write("1/4 — Validating and preparing data")
            progress_bar.progress(20)

            status.write("2/4 — Running ML prediction")
            progress_bar.progress(45)

            status.write("3/4 — Retrieving guidelines and generating reasoning")
            progress_bar.progress(75)

            result = run_agent_workflow(
                raw_data=uploaded_df.to_dict(orient="records"),
                selected_station=selected_station,
                model=model,
            )

            progress_bar.progress(100)
            status.update(label="Workflow completed successfully", state="complete")

            st.markdown("---")
            render_forecast_section(result)
            render_reasoning_section(result)
            render_plan_section(result)

            if st.session_state.get("show_raw_json", True):
                render_debug_and_export(result)
        except Exception as exc:
            progress_bar.empty()
            status.update(label="Workflow failed", state="error")
            logger.exception("Agent workflow failed: %s", exc)
            st.error("The agent workflow failed. Please check the uploaded data and environment setup.")
            st.exception(exc)
        finally:
            try:
                progress_bar.empty()
            except Exception:
                pass


if __name__ == "__main__":
    main()
