from __future__ import annotations

from typing import Any, TypedDict


class EVAgentState(TypedDict, total=False):
    raw_data: list[dict[str, Any]]
    processed_data: list[dict[str, Any]]
    predictions: list[dict[str, Any]]
    hotspots: list[dict[str, Any]]
    retrieved_docs: list[str]
    reasoning: list[str]
    reasoning_summary: str
    final_plan: dict[str, Any]
    summary: dict[str, Any]
    optimization: dict[str, Any]
    data_quality: dict[str, Any]
    selected_station: int
    model: Any
    errors: list[str]
    warnings: list[str]
    rag_fallback_used: bool
    insufficient_data: bool


EVState = EVAgentState
