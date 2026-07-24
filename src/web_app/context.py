"""
Application context: build the agents, the chat assistant (with SQLite-persisted
conversations), and the thread store — once, shared across the UI.

The Chat tab uses `assistant`; the structured tabs (Portfolio/Market/Goals) call
`agents` directly, which is the split the workflow was designed around.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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
from src.web_app.thread_store import ThreadStore, get_checkpointer
from src.workflow.assistant import FinanceAssistant
from src.workflow.nodes import WorkflowNodes
from src.workflow.router import Router

INDEX_DIR = Path(__file__).resolve().parents[1] / "data" / "index"


@dataclass
class AppContext:
    assistant: FinanceAssistant
    agents: dict
    thread_store: ThreadStore


def build_context() -> AppContext:
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
    assistant = FinanceAssistant(nodes, checkpointer=get_checkpointer())
    return AppContext(assistant=assistant, agents=agents, thread_store=ThreadStore())
