"""
Base class shared by all six specialized agents.

Keeps the contract deliberately small: a name, a description (used by the
workflow router to decide which agent handles a query), and an injectable
LLM so every agent is testable with a fake model.
"""

from abc import ABC, abstractmethod
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

EDUCATIONAL_DISCLAIMER = (
    "This is educational information, not financial advice. Consider consulting "
    "a licensed financial advisor before making investment decisions."
)


class BaseFinanceAgent(ABC):
    """Common surface for all finance agents.

    Subclasses implement `run`, which takes agent-specific typed input and
    returns agent-specific typed output. The LangGraph workflow layer adapts
    graph state to these typed calls (thin adapter, added in the
    orchestration milestone).
    """

    name: str
    description: str

    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Execute this agent's task."""
        raise NotImplementedError
