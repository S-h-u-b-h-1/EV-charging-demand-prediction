from rag.vectorstore import retrieve_guidelines


# -------------------------
# DEMAND ANALYSIS
# -------------------------
def demand_analysis_node(state):
    df = state["predictions"]

    state["avg_demand"] = df["predicted_demand"].mean()
    state["peak_demand"] = df["predicted_demand"].max()

    return state


# -------------------------
# HOTSPOT DETECTION
# -------------------------
def hotspot_detection_node(state):
    df = state["predictions"]

    hotspots = df[df["predicted_demand"] > 25]

    if "station_id" in df.columns:
        state["hotspots"] = hotspots["station_id"].astype(str).tolist()
    else:
        state["hotspots"] = hotspots.index.astype(str).tolist()

    return state


# -------------------------
# RAG RETRIEVAL
# -------------------------
def rag_retrieval_node(state):
    query = "EV charging infrastructure planning for high demand zones"

    docs = retrieve_guidelines(query)

    state["retrieved_docs"] = docs

    return state


# -------------------------
# REASONING NODE
# -------------------------
def reasoning_node(state):
    hotspots = state.get("hotspots", [])
    docs = state.get("retrieved_docs", [])

    state["insights"] = docs

    state["recommendations"] = [
        f"Install fast chargers in zones: {hotspots}",
        "Implement load balancing strategies",
        "Introduce time-of-use pricing",
    ]

    return state


# -------------------------
# OPTIMIZATION NODE
# -------------------------
def optimization_node(state):

    state["schedule"] = [
        "Peak hours (6PM–11PM): prioritize fast charging",
        "Off-peak hours: offer discounted pricing",
        "Distribute load evenly across stations"
    ]

    return state


# -------------------------
# FINAL OUTPUT FORMATTER
# -------------------------
def output_formatter_node(state):

    return {
        "high_load_zones": state.get("hotspots", []),
        "insights": state.get("insights", []),
        "recommendations": state.get("recommendations", []),
        "scheduling_plan": state.get("schedule", [])
    }