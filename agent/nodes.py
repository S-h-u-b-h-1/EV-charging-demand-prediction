import pandas as pd

from rag.vectorstore import retrieve_guidelines


# -------------------------
# DEMAND ANALYSIS NODE
# -------------------------
def demand_analysis_node(state):
    df = pd.DataFrame(state["predictions"])

    state["avg_demand"] = float(df["predicted_demand"].mean())
    state["peak_demand"] = float(df["predicted_demand"].max())
    state["peak_hour"] = str(df.loc[df["predicted_demand"].idxmax(), "hour"])

    return state


# -------------------------
# HOTSPOT DETECTION NODE (IMPROVED)
# -------------------------
def hotspot_detection_node(state):
    df = pd.DataFrame(state["predictions"])

    threshold = df["predicted_demand"].mean() + df["predicted_demand"].std()
    hotspots_df = df[df["predicted_demand"] > threshold].copy()

    # rank hotspots
    hotspots_df = hotspots_df.sort_values("predicted_demand", ascending=False)

    hotspots_df["severity"] = hotspots_df["predicted_demand"]

    hotspots = []
    hotspot_details = []

    for _, row in hotspots_df.head(8).iterrows():
        hotspots.append(str(row["hour"]))
        hotspot_details.append({
            "hour": str(row["hour"]),
            "demand": float(row["predicted_demand"]),
            "severity": float(row["predicted_demand"])
        })

    state["hotspots"] = hotspots
    state["hotspot_details"] = hotspot_details

    return state


# -------------------------
# RAG RETRIEVAL NODE
# -------------------------
def rag_retrieval_node(state):
    query = "EV charging infrastructure optimization, load balancing, peak demand management"
    docs = retrieve_guidelines(query)

    state["retrieved_docs"] = docs
    return state


# -------------------------
# REASONING NODE (UPGRADED AGENT LOGIC)
# -------------------------
def reasoning_node(state):
    hotspots = state.get("hotspots", [])
    peak = float(state.get("peak_demand", 0))

    # adaptive charger planning (range instead of fixed number)
    if peak > 30:
        charger_plan = "3–5 fast chargers"
    elif peak > 20:
        charger_plan = "2–4 fast chargers"
    else:
        charger_plan = "1–2 fast chargers"

    # risk scoring
    if peak > 30:
        risk = "Critical"
    elif peak > 25:
        risk = "High"
    elif peak > 15:
        risk = "Medium"
    else:
        risk = "Low"

    # load pattern insight
    concentration = "highly concentrated" if len(hotspots) <= 3 else "distributed"

    state["risk_level"] = risk

    state["insights"] = [
        f"Peak demand observed: {peak:.2f} kWh",
        f"Peak load concentration: {concentration}",
        f"Recommended infrastructure: {charger_plan}"
    ]

    state["recommendations"] = [
        f"Deploy {charger_plan} based on demand intensity and load variability",
        f"Prioritize load balancing during peak hour {state.get('peak_hour', 'N/A')}",
        "Enable dynamic pricing to flatten demand curve",
        "Monitor hotspot clusters for real-time scaling"
    ]

    return state


# -------------------------
# OPTIMIZATION NODE
# -------------------------
def optimization_node(state):
    hotspots = state.get("hotspots", [])

    peak_window = ", ".join(hotspots[:5]) if hotspots else "N/A"

    state["schedule"] = [
        f"Peak window ({peak_window}): prioritize fast charging allocation",
        "Off-peak hours: incentivize discounted charging",
        "Distribute load dynamically across nearby stations",
        "Enable predictive scaling for next-day demand shifts"
    ]

    return state


# -------------------------
# FINAL OUTPUT FORMATTER (CLEAN + PROFESSIONAL)
# -------------------------
def output_formatter_node(state):
    return {
        "summary": {
            "avg_demand": state.get("avg_demand"),
            "peak_demand": state.get("peak_demand"),
            "peak_hour": state.get("peak_hour"),
            "risk_level": state.get("risk_level")
        },

        "hotspots": state.get("hotspot_details", []),

        "insights": state.get("insights", []),

        "recommendations": state.get("recommendations", []),

        "scheduling_plan": state.get("schedule", []),

        "knowledge_sources": state.get("retrieved_docs", [])
    }