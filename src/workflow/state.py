"""
Shared conversation state for the LangGraph workflow.

The state flows through the graph: the router reads `query` and sets `route`;
an agent node fills `response` (the text reply) and `result_data` (the structured
result, for the UI to render). `messages` accumulates the conversation via the
add_messages reducer, and `pending_portfolio` carries a draft across turns so the
portfolio flow can echo-back and confirm before analyzing.
"""

from typing import Annotated, Optional, TypedDict

from langgraph.graph.message import add_messages

# The categories the router chooses between (also the agent-node names).
VALID_ROUTES = [
    "portfolio_analysis",
    "tax_education",
    "goal_planning",
    "market_analysis",
    "finance_qa",
    "clarify",
]


class AgentState(TypedDict, total=False):
    """Graph state. `total=False` so nodes can return partial updates."""

    messages: Annotated[list, add_messages]  # full conversation history
    query: str                               # the latest user message
    route: str                               # chosen agent (set by router)
    response: str                            # text reply shown to the user
    result_data: Optional[dict]              # structured result for the UI
    knowledge_level: str                     # "beginner" | "intermediate"
    pending_portfolio: Optional[list]        # draft holdings awaiting confirmation
    error: Optional[str]
