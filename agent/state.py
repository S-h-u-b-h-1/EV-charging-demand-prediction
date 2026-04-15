from typing import TypedDict, List
import pandas as pd

class EVState(TypedDict):
    predictions: pd.DataFrame
    avg_demand: float
    peak_demand: float
    hotspots: List[str]
    retrieved_docs: List[str]
    insights: List[str]
    recommendations: List[str]
    schedule: List[str]