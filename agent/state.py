from typing import TypedDict, List, Dict

class EVState(TypedDict):
    predictions: List[Dict]   # ✅ FIX (was DataFrame)
    avg_demand: float
    peak_demand: float
    hotspots: List[str]
    retrieved_docs: List[str]
    insights: List[str]
    recommendations: List[str]
    schedule: List[str]