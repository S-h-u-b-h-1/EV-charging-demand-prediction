from langgraph.graph import StateGraph, END
from agent.nodes import (
    demand_analysis_node,
    hotspot_detection_node,
    rag_retrieval_node,
    reasoning_node,
    optimization_node,
    output_formatter_node
)
from agent.state import EVState


def build_graph():
    workflow = StateGraph(EVState)

    # Nodes
    workflow.add_node("demand_analysis", demand_analysis_node)
    workflow.add_node("hotspot_detection", hotspot_detection_node)
    workflow.add_node("rag_retrieval", rag_retrieval_node)
    workflow.add_node("reasoning", reasoning_node)
    workflow.add_node("optimization", optimization_node)
    workflow.add_node("output_formatter", output_formatter_node)

    # Flow
    workflow.set_entry_point("demand_analysis")
    workflow.add_edge("demand_analysis", "hotspot_detection")
    workflow.add_edge("hotspot_detection", "rag_retrieval")
    workflow.add_edge("rag_retrieval", "reasoning")
    workflow.add_edge("reasoning", "optimization")
    workflow.add_edge("optimization", "output_formatter")
    workflow.add_edge("output_formatter", END)

    return workflow.compile()


def run_agent(input_state: dict):
    graph = build_graph()
    result = graph.invoke(input_state)
    return result