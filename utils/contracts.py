ARCHITECTURE_TEXT = """CSV Input
  -> Preprocessing
  -> ML Model
  -> Forecast
  -> LangGraph Agent
  -> RAG
  -> Planning Engine
  -> UI"""

WORKFLOW_TEXT = """Input Node
  -> Preprocessing Node
  -> Prediction Node
  -> Hotspot Detection Node
  -> RAG Retrieval Node
  -> Reasoning Node
  -> Planning Node
  -> Output Node"""

INPUT_SPEC = [
    "CSV file containing station_id, timestamp/hour, and model-ready demand features.",
    "Minimum operational fields expected by the app: station_encoded and hour.",
    "Optional trend and calendar features can be imputed when partially missing.",
]

OUTPUT_SPEC = [
    "predictions",
    "avg_demand",
    "peak_demand",
    "peak_hour",
    "hotspots",
    "infrastructure_plan",
    "schedule",
    "recommendations",
    "reasoning",
    "errors/warnings",
]
