"""
FinanceAssistant: the single entry point the UI (and demos) call.

Wraps the compiled LangGraph workflow. `chat(message, thread_id)` runs one turn
and returns the text reply plus the structured result and which agent handled it.
`build_default_assistant()` wires the real agents, retriever, and data services.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from langchain_core.messages import HumanMessage

from src.workflow.graph import build_finance_graph
from src.workflow.nodes import WorkflowNodes
from src.workflow.router import Router

INDEX_DIR = Path(__file__).resolve().parents[1] / "data" / "index"


@dataclass
class AssistantReply:
    """One turn's reply: the text, which agent handled it, and structured data."""
    text: str
    route: str
    data: Optional[dict] = None


class FinanceAssistant:
    """Thin facade over the compiled workflow graph."""

    def __init__(self, nodes: WorkflowNodes, checkpointer=None):
        self.graph = build_finance_graph(nodes, checkpointer=checkpointer)

    def chat(self, message: str, thread_id: str = "default",
             knowledge_level: str = "beginner") -> AssistantReply:
        state = self.graph.invoke(
            {"messages": [HumanMessage(content=message)],
             "query": message,
             "knowledge_level": knowledge_level},
            config={"configurable": {"thread_id": thread_id}},
        )
        return AssistantReply(
            text=state.get("response", ""),
            route=state.get("route", "clarify"),
            data=state.get("result_data"),
        )


def build_default_assistant(checkpointer=None) -> FinanceAssistant:
    """Wire the real agents and services. Requires OPENAI_API_KEY and a prebuilt
    index (scripts/build_index.py); ALPHAVANTAGE_API_KEY is optional (news).

    Pass a `checkpointer` (e.g. SqliteSaver) to persist conversations; defaults
    to in-memory (MemorySaver) when None."""
    from dotenv import load_dotenv

    from src.agents.finance_qa.agent import FinanceQAAgent
    from src.agents.goal_planning.agent import GoalPlanningAgent
    from src.agents.market_analysis.agent import MarketAnalysisAgent
    from src.agents.portfolio_analysis.agent import PortfolioAnalysisAgent
    from src.agents.tax_education.agent import TaxEducationAgent
    from src.core.llm import get_llm
    from src.data.market_analysis_service import MarketAnalysisService
    from src.data.market_data import MarketDataService
    from src.data.news_client import AlphaVantageNewsClient
    from src.rag.embedder import OpenAIEmbedder
    from src.rag.retriever import HybridRetriever
    from src.rag.vector_store import VectorStore

    load_dotenv()
    llm = get_llm()
    retriever = HybridRetriever(VectorStore.load(INDEX_DIR), OpenAIEmbedder())

    agents = {
        "finance_qa": FinanceQAAgent(llm, retriever),
        "portfolio_analysis": PortfolioAnalysisAgent(llm, MarketDataService()),
        "market_analysis": MarketAnalysisAgent(
            llm, MarketAnalysisService(), retriever=retriever,
            news_client=AlphaVantageNewsClient(),
        ),
        "goal_planning": GoalPlanningAgent(llm, retriever=retriever),
        "tax_education": TaxEducationAgent(llm, retriever),
    }
    nodes = WorkflowNodes(router=Router(llm), agents=agents, extractor_llm=llm)
    return FinanceAssistant(nodes, checkpointer=checkpointer)
