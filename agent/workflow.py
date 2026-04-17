from __future__ import annotations

import math
from typing import Any

import pandas as pd
from langgraph.graph import END, StateGraph

from agent.state import EVAgentState
from ml.config import MODEL_FEATURES
from ml.inference import PREDICTION_FAILURE_MESSAGE, run_prediction
from rag.vectorstore import retrieve_guidelines
from utils.logger import get_logger
from utils.validation import build_data_quality_summary, prepare_feature_frame

logger = get_logger(__name__)

RAG_FALLBACK_MESSAGE = "No planning guidelines retrieved. Using default EV infrastructure rules."
INSUFFICIENT_DATA_MESSAGE = "Insufficient data for reliable planning"


def input_node(state: EVAgentState) -> EVAgentState:
    raw_records = state.get("raw_data", [])
    if not raw_records:
        state.setdefault("errors", []).append("No input data was supplied to the agent workflow.")
        state["insufficient_data"] = True
        return state

    raw_df = pd.DataFrame(raw_records)
    selected_station = state.get("selected_station")
    if "station_encoded" in raw_df.columns:
        raw_df["station_encoded"] = pd.to_numeric(raw_df["station_encoded"], errors="coerce")
        raw_df = raw_df.dropna(subset=["station_encoded"])
        if not raw_df.empty:
            raw_df["station_encoded"] = raw_df["station_encoded"].astype(int)

    if selected_station is not None and "station_encoded" in raw_df.columns:
        station_df = raw_df[raw_df["station_encoded"] == selected_station].copy()
    else:
        station_df = raw_df.copy()

    if station_df.empty:
        state.setdefault("errors", []).append("Selected station does not contain any usable rows.")
        state["insufficient_data"] = True
    else:
        state["raw_data"] = station_df.to_dict(orient="records")

    return state


def preprocessing_node(state: EVAgentState) -> EVAgentState:
    raw_df = pd.DataFrame(state.get("raw_data", []))
    if raw_df.empty:
        state["processed_data"] = []
        state["data_quality"] = {"insufficient_data": True, "insufficient_reasons": ["No rows available after input filtering."]}
        return state

    processed_df, warnings = prepare_feature_frame(raw_df)
    state.setdefault("warnings", []).extend(warnings)
    state["processed_data"] = processed_df.to_dict(orient="records")
    state["data_quality"] = build_data_quality_summary(processed_df, state.get("warnings", []))

    if state["data_quality"].get("insufficient_data"):
        state["insufficient_data"] = True

    return state


def prediction_node(state: EVAgentState) -> EVAgentState:
    processed_df = pd.DataFrame(state.get("processed_data", []))
    model = state.get("model")
    if processed_df.empty or model is None:
        state.setdefault("errors", []).append(PREDICTION_FAILURE_MESSAGE)
        state["predictions"] = []
        state["insufficient_data"] = True
        return state

    predictions, prediction_error = run_prediction(model, processed_df)
    if prediction_error:
        state.setdefault("errors", []).append(prediction_error)
        state["predictions"] = []
        state["insufficient_data"] = True
        return state

    processed_df["predicted_demand"] = predictions
    state["predictions"] = processed_df[["hour", "station_encoded", "predicted_demand"]].to_dict(orient="records")
    return state


def hotspot_detection_node(state: EVAgentState) -> EVAgentState:
    prediction_df = pd.DataFrame(state.get("predictions", []))
    if prediction_df.empty:
        state["hotspots"] = []
        return state

    hourly = prediction_df.groupby("hour")["predicted_demand"].mean().reset_index()
    std_value = float(hourly["predicted_demand"].std()) if len(hourly) > 1 else 0.0
    threshold = float(hourly["predicted_demand"].mean()) + (0.0 if pd.isna(std_value) else std_value)
    hotspots_df = hourly[hourly["predicted_demand"] > threshold].sort_values("predicted_demand", ascending=False)

    state["hotspots"] = [
        {
            "hour": f"{int(row['hour'])}:00",
            "demand": round(float(row["predicted_demand"]), 2),
            "severity": "High",
        }
        for _, row in hotspots_df.head(5).iterrows()
    ]
    return state


def rag_retrieval_node(state: EVAgentState) -> EVAgentState:
    prediction_df = pd.DataFrame(state.get("predictions", []))
    peak_demand = float(prediction_df["predicted_demand"].max()) if not prediction_df.empty else 0.0
    query = (
        "EV charging infrastructure planning for hourly demand forecast. "
        f"Peak demand {peak_demand:.2f} kWh. Include charger utilization, load balancing, "
        "grid constraints, and cost-performance trade-offs."
    )
    docs = retrieve_guidelines(query)
    if docs:
        state["retrieved_docs"] = docs
        state["rag_fallback_used"] = False
    else:
        logger.warning("RAG retrieval failed or returned no docs; enabling fallback guidance.")
        state["retrieved_docs"] = [RAG_FALLBACK_MESSAGE]
        state["rag_fallback_used"] = True
        state.setdefault("warnings", []).append(RAG_FALLBACK_MESSAGE)
    return state


def reasoning_node(state: EVAgentState) -> EVAgentState:
    prediction_df = pd.DataFrame(state.get("predictions", []))
    data_quality = state.get("data_quality", {})

    if prediction_df.empty or data_quality.get("insufficient_data"):
        reasons = data_quality.get("insufficient_reasons", [])
        state["insufficient_data"] = True
        state["reasoning"] = [INSUFFICIENT_DATA_MESSAGE] + reasons
        return state

    hourly = prediction_df.groupby("hour")["predicted_demand"].mean().reset_index()
    avg_demand = float(prediction_df["predicted_demand"].mean())
    peak_row = hourly.loc[hourly["predicted_demand"].idxmax()]
    peak_demand = float(peak_row["predicted_demand"])
    peak_hour = int(peak_row["hour"])

    charger_type, nominal_capacity, charger_cost = _select_charger_profile(peak_demand)
    chosen_count, chosen_utilization, alternatives = _optimize_charger_count(peak_demand, nominal_capacity)
    load_balancing_required = peak_demand > 25
    grid_status = "Load balancing required to avoid transformer overload." if load_balancing_required else "Grid load remains within normal operating threshold."
    smoothing_target = max(avg_demand, peak_demand * 0.85)

    reasoning = [
        f"Peak demand is {peak_demand:.2f} kWh at {peak_hour}:00 with average demand {avg_demand:.2f} kWh.",
        f"Selected {chosen_count} {charger_type} because peak-time utilization is {chosen_utilization * 100:.1f}%, which stays near the 70-90% target band.",
        f"Using fewer chargers would push utilization above {alternatives['fewer_utilization'] * 100:.1f}% and risk queuing or overload." if alternatives["fewer_utilization"] is not None else "A lower charger count would not safely absorb the peak hour.",
        f"Adding more chargers would drop utilization to {alternatives['more_utilization'] * 100:.1f}%, increasing idle infrastructure cost." if alternatives["more_utilization"] is not None else "The selected charger count already sits at the practical lower-cost boundary.",
        f"Peak load smoothing target is {smoothing_target:.2f} kWh, so scheduling should shift discretionary demand away from the highest-load window.",
        grid_status,
    ]

    state["reasoning"] = reasoning
    state["summary"] = {
        "avg_demand": avg_demand,
        "peak_demand": peak_demand,
        "peak_hour": f"{peak_hour}:00",
        "risk_level": _risk_level(peak_demand),
    }
    state["optimization"] = {
        "charger_type": charger_type,
        "charger_capacity_kwh": nominal_capacity,
        "charger_count": chosen_count,
        "target_utilization": chosen_utilization,
        "grid_constraint_triggered": load_balancing_required,
        "cost_band": charger_cost,
        "peak_hour": peak_hour,
        "avg_demand": avg_demand,
    }
    state["reasoning_summary"] = " ".join(reasoning)
    return state


def planning_node(state: EVAgentState) -> EVAgentState:
    if state.get("insufficient_data"):
        safe_plan = {
            "infrastructure_plan": [
                {
                    "station_id": state.get("selected_station", "unknown"),
                    "chargers": 0,
                    "charger_type": "Deferred",
                    "utilization_target": "N/A",
                    "load_balancing": "Reassess after collecting more demand history",
                }
            ],
            "schedule": ["Collect more demand data before applying an automated charging schedule."],
            "recommendations": [INSUFFICIENT_DATA_MESSAGE],
            "explanation": INSUFFICIENT_DATA_MESSAGE,
        }
        state["final_plan"] = safe_plan
        return state

    optimization = state.get("optimization", {})
    prediction_df = pd.DataFrame(state.get("predictions", []))
    hourly = prediction_df.groupby("hour")["predicted_demand"].mean().reset_index()
    peak_hour = int(optimization["peak_hour"])
    peak_hours = sorted({hour for hour in [peak_hour - 1, peak_hour, peak_hour + 1] if 0 <= hour <= 23})
    off_peak_hours = _select_off_peak_hours(hourly, peak_hours, limit=6)

    infrastructure_plan = [
        {
            "station_id": state.get("selected_station", "unknown"),
            "chargers": optimization["charger_count"],
            "charger_type": optimization["charger_type"],
            "utilization_target": f"{optimization['target_utilization'] * 100:.1f}%",
            "load_balancing": "Required" if optimization["grid_constraint_triggered"] else "Not required",
            "cost_band": optimization["cost_band"],
        }
    ]

    schedule = [
        f"Peak window ({', '.join(f'{hour}:00' for hour in peak_hours)}): prioritize queue control and defer flexible charging.",
        f"Off-peak window ({', '.join(f'{hour}:00' for hour in off_peak_hours)}): incentivize discounted charging and background fleet charging.",
        "Apply smart scheduling to shift 10-15% of discretionary demand from peak to off-peak hours.",
    ]

    recommendations = [
        f"Deploy {optimization['charger_count']} {optimization['charger_type']} units for balanced utilization and acceptable queue risk.",
        "Enable load balancing controls whenever predicted demand crosses the 25 kWh grid threshold." if optimization["grid_constraint_triggered"] else "Keep monitoring transformer loading; dynamic balancing can remain on standby.",
        "Use time-of-use pricing and reservation nudges to smooth demand before the peak hour.",
    ]

    explanation = (
        f"The plan chooses {optimization['charger_count']} chargers instead of a simple peak/10 rule because it optimizes "
        f"for 70-90% utilization, manages grid risk, and avoids overspending on idle infrastructure. "
        f"Peak demand is served with {optimization['target_utilization'] * 100:.1f}% expected utilization."
    )

    state["final_plan"] = {
        "infrastructure_plan": infrastructure_plan,
        "schedule": schedule,
        "recommendations": recommendations,
        "explanation": explanation,
    }
    return state


def output_node(state: EVAgentState) -> dict[str, Any]:
    return {
        "workflow": "This system uses LangGraph-based stateful agent workflow.",
        "raw_data": state.get("raw_data", []),
        "processed_data": state.get("processed_data", []),
        "predictions": state.get("predictions", []),
        "hotspots": state.get("hotspots", []),
        "retrieved_docs": state.get("retrieved_docs", []),
        "reasoning": state.get("reasoning", []),
        "reasoning_summary": state.get("reasoning_summary", ""),
        "final_plan": state.get("final_plan", {}),
        "summary": state.get("summary", {}),
        "errors": list(dict.fromkeys(state.get("errors", []))),
        "warnings": list(dict.fromkeys(state.get("warnings", []))),
        "rag_fallback_used": state.get("rag_fallback_used", False),
        "insufficient_data": state.get("insufficient_data", False),
    }


def build_workflow():
    workflow = StateGraph(EVAgentState)
    workflow.add_node("input", input_node)
    workflow.add_node("preprocessing", preprocessing_node)
    workflow.add_node("prediction", prediction_node)
    workflow.add_node("hotspot_detection", hotspot_detection_node)
    workflow.add_node("rag_retrieval", rag_retrieval_node)
    workflow.add_node("reasoning", reasoning_node)
    workflow.add_node("planning", planning_node)
    workflow.add_node("output", output_node)

    workflow.set_entry_point("input")
    workflow.add_edge("input", "preprocessing")
    workflow.add_edge("preprocessing", "prediction")
    workflow.add_edge("prediction", "hotspot_detection")
    workflow.add_edge("hotspot_detection", "rag_retrieval")
    workflow.add_edge("rag_retrieval", "reasoning")
    workflow.add_edge("reasoning", "planning")
    workflow.add_edge("planning", "output")
    workflow.add_edge("output", END)
    return workflow.compile()


def _select_charger_profile(peak_demand: float) -> tuple[str, float, str]:
    if peak_demand > 25:
        return "DC Fast Charger (50kW+)", 12.0, "High capex / high service speed"
    if peak_demand > 15:
        return "Hybrid AC/DC Charger", 8.0, "Balanced capex / balanced flexibility"
    return "Level 2 AC Charger", 6.0, "Lower capex / moderate service speed"


def _optimize_charger_count(peak_demand: float, nominal_capacity: float) -> tuple[int, float, dict[str, float | None]]:
    target_low = 0.70
    target_high = 0.90
    candidates = []

    max_count = max(2, int(math.ceil(max(peak_demand, nominal_capacity) / nominal_capacity)) + 3)
    for count in range(1, max_count + 1):
        utilization = peak_demand / (count * nominal_capacity) if count > 0 and nominal_capacity > 0 else 0.0
        within_band = target_low <= utilization <= target_high
        distance = 0.0 if within_band else min(abs(utilization - target_low), abs(utilization - target_high))
        cost_penalty = count * 0.01
        candidates.append((distance + cost_penalty, count, utilization))

    _, chosen_count, chosen_utilization = min(candidates, key=lambda item: item[0])

    fewer_utilization = None
    if chosen_count > 1:
        fewer_utilization = peak_demand / ((chosen_count - 1) * nominal_capacity)

    more_utilization = peak_demand / ((chosen_count + 1) * nominal_capacity)

    return chosen_count, chosen_utilization, {
        "fewer_utilization": fewer_utilization,
        "more_utilization": more_utilization,
    }


def _risk_level(peak_demand: float) -> str:
    if peak_demand > 25:
        return "High"
    if peak_demand > 15:
        return "Moderate"
    return "Low"


def _select_off_peak_hours(hourly: pd.DataFrame, peak_hours: list[int], limit: int) -> list[int]:
    if hourly.empty:
        base_hours = [0, 1, 2, 3, 4, 5]
        return [hour for hour in base_hours if hour not in peak_hours][:limit]

    off_peak = (
        hourly[~hourly["hour"].isin(peak_hours)]
        .sort_values("predicted_demand", ascending=True)["hour"]
        .astype(int)
        .tolist()
    )

    if len(off_peak) < limit:
        fallback_hours = [hour for hour in range(24) if hour not in peak_hours and hour not in off_peak]
        off_peak.extend(fallback_hours)

    return off_peak[:limit]
