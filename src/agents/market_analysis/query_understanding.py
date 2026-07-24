"""
Query understanding: natural language -> (intent, tickers, period).

Primary path uses the LLM to return strict JSON. If the model is unavailable
or returns unparseable output, a keyword/regex heuristic keeps the agent
working (degraded but functional) — the same fail-soft philosophy used
elsewhere in the codebase.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.market_analysis.model import AnalysisType, Period

logger = logging.getLogger(__name__)

# Major indices used for the "how's the market" overview intent.
MARKET_INDICES = ["^GSPC", "^IXIC", "^DJI"]  # S&P 500, Nasdaq, Dow

_EXTRACTION_PROMPT = """You extract structured intent from a user's market \
question. Respond with ONLY a JSON object, no prose, with these keys:
- "intent": one of snapshot, metric, trend, compare, overview, news, concept
- "tickers": list of stock ticker symbols mentioned (uppercase). [] if none.
- "period": one of 1d, 5d, 1mo, 6mo, 1y, 5y (best guess; default 1mo)
- "ambiguous": true if a company is named but you're unsure of its ticker

Intent guide:
- snapshot: current price / how a stock is doing right now
- metric: a specific number like P/E, dividend yield, beta, market cap
- trend: performance over time (this week/month/year)
- compare: two or more tickers weighed against each other
- overview: the market/indices in general, no specific company
- news: news or sentiment about a company
- concept: a general question about how markets work (no specific data lookup)

Example: "How has Apple done this year vs Microsoft?" ->
{"intent":"compare","tickers":["AAPL","MSFT"],"period":"1y","ambiguous":false}"""

_PERIOD_VALUES = {p.value for p in Period}
_INTENT_VALUES = {a.value for a in AnalysisType}

# Words that look like tickers but aren't, to reduce heuristic false positives.
_TICKER_STOPWORDS = {
    "A", "I", "PE", "EPS", "ETF", "CEO", "IPO", "USA", "US", "AI", "P", "E",
    "THE", "AND", "OR", "VS", "IS", "IT", "TO", "OF", "MY", "DO",
}


@dataclass
class ExtractedQuery:
    intent: AnalysisType
    tickers: list[str] = field(default_factory=list)
    period: Period = Period.ONE_MONTH
    ambiguous: bool = False


class QueryUnderstanding:
    """Turns a raw question into a structured ExtractedQuery."""

    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    def extract(self, question: str) -> ExtractedQuery:
        try:
            response = self.llm.invoke(
                [SystemMessage(content=_EXTRACTION_PROMPT),
                 HumanMessage(content=question)]
            )
            parsed = self._parse(response.content)
            if parsed is not None:
                return parsed
            logger.info("Extraction JSON unparseable; heuristic fallback")
        except Exception:
            logger.warning("LLM extraction failed; using heuristic", exc_info=True)
        # Always run the heuristic on the *question*, never on the LLM output.
        return self._heuristic(question)

    def _parse(self, raw: str) -> Optional[ExtractedQuery]:
        """Parse the LLM's JSON. Returns None (not a guess) if it isn't valid
        JSON, so the caller can fall back to the heuristic on the question."""
        text = raw.strip()
        # Strip ```json fences if present.
        text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(data, dict):
            return None

        intent_raw = str(data.get("intent", "")).lower()
        intent = AnalysisType(intent_raw) if intent_raw in _INTENT_VALUES else AnalysisType.SNAPSHOT
        period_raw = str(data.get("period", "")).lower()
        period = Period(period_raw) if period_raw in _PERIOD_VALUES else Period.ONE_MONTH
        tickers = [str(t).strip().upper() for t in data.get("tickers", []) if str(t).strip()]
        return ExtractedQuery(
            intent=intent, tickers=tickers, period=period,
            ambiguous=bool(data.get("ambiguous", False)),
        )

    @staticmethod
    def _heuristic(question: str) -> ExtractedQuery:
        q = question.lower()

        # Tickers: $-prefixed, or bare 1-5 letter uppercase tokens not in stoplist.
        tickers: list[str] = []
        for m in re.findall(r"\$([A-Za-z]{1,5})", question):
            tickers.append(m.upper())
        for m in re.findall(r"\b([A-Z]{1,5})\b", question):
            if m not in _TICKER_STOPWORDS and m not in tickers:
                tickers.append(m)

        # Intent by keyword.
        if any(w in q for w in ["compare", " vs ", " versus "]):
            intent = AnalysisType.COMPARE
        elif any(w in q for w in ["news", "sentiment", "headline"]):
            intent = AnalysisType.NEWS
        elif any(w in q for w in ["p/e", "pe ratio", "dividend", "beta", "market cap",
                                   "valuation", "eps", "book value"]):
            intent = AnalysisType.METRIC
        elif any(w in q for w in ["trend", "performance", "over the", "this year",
                                   "this month", "past", "last month", "last year",
                                   "ytd", "how has", "how did"]):
            intent = AnalysisType.TREND
        elif any(w in q for w in ["market", "s&p", "nasdaq", "dow", "indices", "index"]) and not tickers:
            intent = AnalysisType.OVERVIEW
        elif any(w in q for w in ["what moves", "why do", "how do", "what causes",
                                   "what is", "explain"]) and not tickers:
            intent = AnalysisType.CONCEPT
        elif tickers:
            intent = AnalysisType.SNAPSHOT
        else:
            intent = AnalysisType.CONCEPT

        # Period by keyword.
        if "5 year" in q or "5y" in q or "five year" in q:
            period = Period.FIVE_YEAR
        elif "year" in q or "1y" in q or "ytd" in q:
            period = Period.ONE_YEAR
        elif "6 month" in q or "six month" in q:
            period = Period.SIX_MONTH
        elif "week" in q or "5 day" in q:
            period = Period.FIVE_DAY
        elif "today" in q or "1 day" in q:
            period = Period.ONE_DAY
        else:
            period = Period.ONE_MONTH

        return ExtractedQuery(intent=intent, tickers=tickers, period=period, ambiguous=False)
