"""
Query router: classify a user message to one of the specialized agents.

Primary path is the LLM returning a single category label. If the LLM is
unavailable or returns something unrecognized, a keyword heuristic decides —
the same fail-soft pattern used across the agents. Never raises: worst case it
routes to `clarify`.
"""

from __future__ import annotations

import logging

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.workflow.state import VALID_ROUTES

logger = logging.getLogger(__name__)

_ROUTER_PROMPT = """Classify the user's message into exactly one category. \
Respond with ONLY the category word — nothing else.

Categories:
- portfolio_analysis: analyzing THEIR OWN holdings (mentions owning shares, "my \
portfolio", the allocation/diversification of what they hold)
- market_analysis: a specific stock's price, quote, trend, valuation, or news, or \
the overall market/indices
- goal_planning: saving toward a goal, retirement projections, how much to save, \
risk tolerance, or asset-allocation frameworks
- tax_education: taxes, tax-advantaged accounts (401k, IRA, Roth, HSA, 529), or \
how capital gains and dividends are taxed
- finance_qa: general investing education (what a stock/bond/ETF is, \
diversification, compound interest, dollar-cost averaging)
- clarify: greetings, unclear messages, or anything unrelated to finance

Answer with one of: portfolio_analysis, market_analysis, goal_planning, \
tax_education, finance_qa, clarify"""

# Ordered keyword rules for the heuristic fallback. Order encodes priority so
# overlapping terms resolve sensibly (ownership before generic market words).
_HEURISTIC_RULES = [
    ("portfolio_analysis", [
        "my portfolio", "my holdings", "my allocation", "my diversification",
        "analyze my", "i own", "i hold", "shares of", "my positions",
    ]),
    ("tax_education", [
        "tax", "roth", "traditional ira", "401k", "401(k)", " ira", "hsa",
        "529", "capital gain", "dividend tax", "deductible", "deduct",
        "tax-advantaged", "brokerage account",
    ]),
    ("goal_planning", [
        "retire", "retirement", "save for", "saving for", "my goal", "reach my",
        "how much to save", "how much should i save", "afford", "future value",
        "risk tolerance", "modern portfolio", "mpt", "asset allocation", "nest egg",
    ]),
    ("market_analysis", [
        "price", "how is", "how's", "how has", "p/e", "pe ratio", "market",
        "nasdaq", "dow", "s&p", "trend", "quote", "ticker", "news", "sentiment",
        "52-week", "52 week", "earnings", "doing today", "stock",
    ]),
    ("finance_qa", [
        "what is", "what's a", "what are", "explain", "how does", "difference between",
        "compound interest", "dollar cost", "diversification", "index fund",
        "etf", "mutual fund", "bond",
    ]),
]


class Router:
    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    def classify(self, query: str) -> str:
        try:
            resp = self.llm.invoke(
                [SystemMessage(content=_ROUTER_PROMPT), HumanMessage(content=query)]
            )
            label = self._parse_label(resp.content)
            if label:
                return label
            logger.info("Router LLM label unrecognized; heuristic fallback")
        except Exception:
            logger.warning("Router LLM failed; heuristic fallback", exc_info=True)
        return self._heuristic(query)

    @staticmethod
    def _parse_label(text: str) -> str | None:
        t = (text or "").strip().lower()
        for label in VALID_ROUTES:
            first = label.split("_")[0]  # unique per label: portfolio/market/goal/tax/finance/clarify
            if label in t or first in t:
                return label
        return None

    @staticmethod
    def _heuristic(query: str) -> str:
        q = query.lower()
        for route, keywords in _HEURISTIC_RULES:
            if any(k in q for k in keywords):
                return route
        return "clarify"
