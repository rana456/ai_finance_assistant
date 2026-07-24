"""
LangGraph StateGraph wiring.

    START -> router -> (conditional edge on `route`) -> one agent node -> END

Compiled with a checkpointer (MemorySaver by default) so conversation state —
including a pending portfolio draft awaiting confirmation — persists across
turns, keyed by thread_id.
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.workflow.nodes import WorkflowNodes
from src.workflow.state import AgentState

# Router label -> graph node name (they're the same strings).
_AGENT_NODES = [
    "finance_qa",
    "market_analysis",
    "goal_planning",
    "tax_education",
    "portfolio_analysis",
    "clarify",
]


def build_finance_graph(nodes: WorkflowNodes, checkpointer=None):
    """Assemble and compile the workflow graph."""
    graph = StateGraph(AgentState)

    graph.add_node("router", nodes.router_node)
    graph.add_node("finance_qa", nodes.finance_qa_node)
    graph.add_node("market_analysis", nodes.market_analysis_node)
    graph.add_node("goal_planning", nodes.goal_planning_node)
    graph.add_node("tax_education", nodes.tax_education_node)
    graph.add_node("portfolio_analysis", nodes.portfolio_node)
    graph.add_node("clarify", nodes.clarify_node)

    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router",
        nodes.select_route,
        {name: name for name in _AGENT_NODES},
    )
    for name in _AGENT_NODES:
        graph.add_edge(name, END)

    return graph.compile(checkpointer=checkpointer or MemorySaver())
