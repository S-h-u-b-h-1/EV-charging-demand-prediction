from __future__ import annotations

from typing import Any

from agent.workflow import build_workflow
from utils.logger import get_logger

logger = get_logger(__name__)


def build_graph():
    return build_workflow()


def run_agent_workflow(raw_data: list[dict[str, Any]], selected_station: int, model: Any) -> dict[str, Any]:
    graph = build_graph()
    initial_state = {
        "raw_data": raw_data,
        "selected_station": selected_station,
        "model": model,
        "processed_data": [],
        "predictions": [],
        "hotspots": [],
        "retrieved_docs": [],
        "reasoning": [],
        "final_plan": {},
        "summary": {},
        "errors": [],
        "warnings": [],
        "rag_fallback_used": False,
        "insufficient_data": False,
    }

    try:
        final_state = graph.invoke(initial_state)
        return _format_result(final_state)
    except Exception as exc:
        logger.exception("LangGraph workflow execution failed: %s", exc)
        return _format_result({
            "workflow": "This system uses LangGraph-based stateful agent workflow.",
            "raw_data": [],
            "processed_data": [],
            "predictions": [],
            "hotspots": [],
            "retrieved_docs": [],
            "reasoning": ["Insufficient data for reliable planning"],
            "reasoning_summary": "Insufficient data for reliable planning",
            "final_plan": {
                "infrastructure_plan": [],
                "schedule": ["Scheduling plan unavailable because the agent workflow could not complete."],
                "recommendations": ["Insufficient data for reliable planning"],
                "explanation": "Insufficient data for reliable planning",
            },
            "summary": {},
            "errors": ["Agent workflow execution failed. Returning safe fallback response."],
            "warnings": [],
            "rag_fallback_used": False,
            "insufficient_data": True,
        })


def run_agent(input_state: dict[str, Any]) -> dict[str, Any]:
    return run_agent_workflow(
        raw_data=input_state.get("raw_data", []),
        selected_station=input_state.get("selected_station", 0),
        model=input_state.get("model"),
    )


def _format_result(state: dict[str, Any]) -> dict[str, Any]:
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
