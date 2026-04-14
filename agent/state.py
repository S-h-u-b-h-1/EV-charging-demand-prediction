from typing import TypedDict, List, Dict, Any

class EVState(TypedDict):
    location_data: Dict[str, Any]
    demand_forecast: Dict[str, float]
    high_load_zones: List[Dict[str, Any]]
    retrieved_guidelines: List[str]
    recommendations: List[str]
    scheduling_insights: List[str]