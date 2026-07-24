"""
Graph nodes: adapters between conversation state and the specialized agents.

Each node reads the state, builds the agent's typed input, runs the agent, and
writes back a text `response` (for chat) plus `result_data` (for the UI to render
structured output). Agents are injected, so the graph is testable with stubs.

The portfolio node is the one with real logic: it runs the natural-language
extract -> echo-back -> confirm -> analyze flow across turns, using
`pending_portfolio` in the state to remember the draft between messages.
"""

from __future__ import annotations

import logging

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage

from src.agents.finance_qa.model import FinanceQuestion, KnowledgeLevel
from src.agents.goal_planning.model import GoalPlanRequest
from src.agents.market_analysis.model import MarketQuery
from src.agents.portfolio_analysis.model import (
    HoldingInput,
    InputSource,
    PortfolioInput,
)
from src.agents.tax_education.model import TaxQuestion
from src.workflow.extraction import (
    extract_portfolio_holdings,
    is_affirmative,
    is_negative,
)
from src.workflow.router import Router
from src.workflow.state import AgentState

logger = logging.getLogger(__name__)

_CLARIFY_MESSAGE = (
    "I can help with a few things: general investing questions, analyzing a "
    "portfolio you describe, live market data for a stock, planning a savings "
    "goal, and tax-account education. What would you like to explore?"
)


def _reply(state_updates: dict, route: str | None = None) -> dict:
    """Attach an assistant message to the state update for history.

    The route and structured result are stored on the message itself (via
    additional_kwargs) so the UI can render each turn's badge and result card
    correctly even when reopening an old thread."""
    text = state_updates.get("response", "")
    kwargs: dict = {}
    if route:
        kwargs["route"] = route
    if state_updates.get("result_data") is not None:
        kwargs["result_data"] = state_updates["result_data"]
    return {**state_updates, "messages": [AIMessage(content=text, additional_kwargs=kwargs)]}


class WorkflowNodes:
    """Holds the router, the agents, and the extraction LLM; exposes one method
    per graph node."""

    def __init__(self, router: Router, agents: dict, extractor_llm: BaseChatModel):
        self.router = router
        self.agents = agents
        self.extractor_llm = extractor_llm

    # --- routing ---

    def router_node(self, state: AgentState) -> dict:
        # A pending portfolio draft forces the portfolio node so the user's
        # confirmation is handled in context, whatever they typed.
        if state.get("pending_portfolio"):
            return {"route": "portfolio_analysis"}
        return {"route": self.router.classify(state["query"])}

    @staticmethod
    def select_route(state: AgentState) -> str:
        return state.get("route", "clarify")

    def _knowledge_level(self, state: AgentState) -> KnowledgeLevel:
        try:
            return KnowledgeLevel(state.get("knowledge_level", "beginner"))
        except ValueError:
            return KnowledgeLevel.BEGINNER

    # --- conversational agent nodes ---

    def finance_qa_node(self, state: AgentState) -> dict:
        try:
            answer = self.agents["finance_qa"].run(
                FinanceQuestion(question=state["query"],
                                knowledge_level=self._knowledge_level(state))
            )
            return _reply({"response": answer.answer,
                           "result_data": answer.model_dump(mode="json")}, "finance_qa")
        except Exception as e:
            return self._error("finance_qa", e)

    def market_analysis_node(self, state: AgentState) -> dict:
        try:
            result = self.agents["market_analysis"].run(MarketQuery(question=state["query"]))
            return _reply({"response": result.narrative,
                           "result_data": result.model_dump(mode="json")}, "market_analysis")
        except Exception as e:
            return self._error("market_analysis", e)

    def goal_planning_node(self, state: AgentState) -> dict:
        try:
            result = self.agents["goal_planning"].run(GoalPlanRequest(question=state["query"]))
            return _reply({"response": result.narrative,
                           "result_data": result.model_dump(mode="json")}, "goal_planning")
        except Exception as e:
            return self._error("goal_planning", e)

    def tax_education_node(self, state: AgentState) -> dict:
        try:
            answer = self.agents["tax_education"].run(TaxQuestion(question=state["query"]))
            return _reply({"response": answer.answer,
                           "result_data": answer.model_dump(mode="json")}, "tax_education")
        except Exception as e:
            return self._error("tax_education", e)

    def clarify_node(self, state: AgentState) -> dict:
        return _reply({"response": _CLARIFY_MESSAGE, "result_data": None}, "clarify")

    # --- portfolio node: extract -> confirm -> analyze ---

    def portfolio_node(self, state: AgentState) -> dict:
        query = state["query"]
        pending = state.get("pending_portfolio")

        if pending:
            if is_affirmative(query):
                return self._analyze_portfolio(pending)
            if is_negative(query):
                return _reply({
                    "response": "No problem — tell me your holdings again and "
                                "I'll re-read them (e.g. '10 shares of AAPL at $150').",
                    "pending_portfolio": None,
                }, "portfolio_analysis")
            # Neither yes nor no: treat as a restatement and re-extract below.

        extraction = extract_portfolio_holdings(self.extractor_llm, query)
        if not extraction.holdings:
            return _reply({
                "response": "To analyze your portfolio, tell me what you hold — "
                            "for example '10 shares of AAPL at $150 and 5 of MSFT' "
                            "— or enter them in the Portfolio tab.",
                "pending_portfolio": None,
            }, "portfolio_analysis")

        draft = [h.model_dump() for h in extraction.holdings]
        summary = ", ".join(
            f"{h.quantity:g}×{h.ticker}"
            + (f" @ ${h.cost_basis_per_share:g}" if h.cost_basis_per_share else "")
            for h in extraction.holdings
        )
        return _reply({
            "response": f"I read your portfolio as: {summary}. "
                        "Shall I analyze it? (reply 'yes' to confirm)",
            "pending_portfolio": draft,
            "result_data": None,
        }, "portfolio_analysis")

    def _analyze_portfolio(self, pending: list) -> dict:
        try:
            holdings = [HoldingInput(**h) for h in pending]
            portfolio = PortfolioInput(
                holdings=holdings,
                source=InputSource.NATURAL_LANGUAGE,
                confirmed_by_user=True,
            )
            result = self.agents["portfolio_analysis"].run(portfolio)
            return _reply({
                "response": result.narrative,
                "result_data": result.model_dump(mode="json"),
                "pending_portfolio": None,
            }, "portfolio_analysis")
        except Exception as e:
            logger.warning("Portfolio analysis failed", exc_info=True)
            return _reply({
                "response": "I couldn't analyze those holdings — please check the "
                            f"tickers and amounts and try again. ({e})",
                "pending_portfolio": None,
            }, "portfolio_analysis")

    # --- helpers ---

    @staticmethod
    def _error(agent: str, exc: Exception) -> dict:
        logger.warning("Agent %s failed: %s", agent, exc, exc_info=True)
        return _reply({
            "response": "Sorry — something went wrong handling that. Please try "
                        "rephrasing your question.",
            "error": f"{agent}: {exc}",
        }, agent)
