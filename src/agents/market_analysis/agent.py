"""
Market Analysis Agent.

Orchestrates: prediction guard -> query understanding -> route by intent
(data / news / concept) -> deterministic analysis -> LLM narration ->
MarketAnalysisResult (with freshness timestamp, citations, disclaimer).

Guardrails enforced outside the LLM:
- Prediction guard: "will it go up / is it a good buy / price target" style
  questions are refused and redirected to education.
- Freshness: every result carries the data's `as_of` timestamp.
- Grounded concepts: "what moves prices" answers come from the cited RAG
  knowledge base, not model memory.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.base import BaseFinanceAgent, EDUCATIONAL_DISCLAIMER
from src.agents.finance_qa.model import Citation
from src.agents.market_analysis.analyzer import (
    build_analyst_view,
    build_news_sentiment,
    build_snapshot,
    build_trend,
    build_valuation,
)
from src.agents.market_analysis.model import (
    AnalysisType,
    MarketAnalysisResult,
    MarketQuery,
)
from src.agents.market_analysis.query_understanding import (
    MARKET_INDICES,
    ExtractedQuery,
    QueryUnderstanding,
)
from src.data.market_analysis_service import MarketAnalysisService

logger = logging.getLogger(__name__)

# Questions asking us to predict or advise, which we decline.
_PREDICTION_PATTERNS = [
    r"\bwill\b.*\b(go up|go down|rise|fall|drop|crash|moon|rebound|recover)\b",
    r"\bshould i (buy|sell|hold|invest|short)\b",
    r"\bis\s+\w+\s+a (good|bad|smart|safe) (buy|investment|stock|bet)\b",
    r"\b(price )?(target|prediction|forecast)\b.*\b(next|tomorrow|week|month|year)\b",
    r"\bwhat('?s| is| will)\b.*\b(price be|worth be)\b",
    r"\b(is it|now)\s+a good time to (buy|sell)\b",
]

_PREDICTION_REFUSAL = (
    "I can't predict where prices are headed or tell you whether to buy or sell "
    "— that's speculation and personalized advice, neither of which I do. What I "
    "can do is show you the current data and explain what it means: the price and "
    "how it's moved, valuation metrics, trends, and how markets work in general. "
    "Want a snapshot or a trend breakdown instead?"
)

_NARRATION_SYSTEM = """You are a financial education assistant explaining market \
data to a beginner. You are given pre-computed numbers as JSON.

Rules:
- Explain ONLY the numbers provided; never invent or predict figures.
- Define financial terms in plain language (P/E, moving average, volume, etc.).
- Say what a number *signals* in general educational terms, but never tell the \
user to buy, sell, or hold, and never forecast future prices.
- Be warm, clear, and concise (under 250 words).
- Do not write a disclaimer or list sources; those are added automatically."""

_CONCEPT_SYSTEM = """You are a financial education assistant. Answer the user's \
question about how markets work using ONLY the numbered sources provided. Define \
terms simply, stay under 250 words, don't invent facts, and don't give advice or \
predictions. No disclaimer or source list — those are added automatically."""


class MarketAnalysisAgent(BaseFinanceAgent):
    name = "market_analysis"
    description = (
        "Looks up and explains real-time market data — quotes, valuation "
        "metrics, trends, comparisons, market overview, and news sentiment — "
        "plus how-markets-work concepts. No predictions or personalized advice."
    )

    def __init__(
        self,
        llm: BaseChatModel,
        market_data: MarketAnalysisService,
        retriever=None,       # HybridRetriever, for CONCEPT questions (optional)
        news_client=None,     # AlphaVantageNewsClient (optional)
    ):
        super().__init__(llm)
        self.market_data = market_data
        self.retriever = retriever
        self.news_client = news_client
        self.understanding = QueryUnderstanding(llm)

    def run(self, query: MarketQuery) -> MarketAnalysisResult:
        now = datetime.now(timezone.utc)

        # --- Guard: predictions / advice ---
        if self._is_prediction(query.question):
            return self._refusal(query, now)

        extracted = self._resolve_query(query)

        if extracted.intent == AnalysisType.CONCEPT:
            return self._handle_concept(query, now)
        if extracted.intent == AnalysisType.NEWS:
            return self._handle_news(query, extracted, now)
        if extracted.intent == AnalysisType.OVERVIEW:
            return self._handle_data(query, extracted, MARKET_INDICES, now)

        # Remaining data intents need at least one ticker.
        if not extracted.tickers:
            return self._no_ticker(query, now)
        return self._handle_data(query, extracted, extracted.tickers, now)

    # --- query resolution ---

    def _resolve_query(self, query: MarketQuery) -> ExtractedQuery:
        """Honor explicit fields on the query; otherwise extract via LLM."""
        if query.tickers is not None or query.analysis_type is not None:
            extracted = self.understanding.extract(query.question)
            return ExtractedQuery(
                intent=query.analysis_type or extracted.intent,
                tickers=query.tickers if query.tickers is not None else extracted.tickers,
                period=query.period,
                ambiguous=extracted.ambiguous,
            )
        extracted = self.understanding.extract(query.question)
        # A user-set period on the query overrides the guessed one.
        if query.period is not None:
            extracted.period = query.period
        return extracted

    # --- intent handlers ---

    def _handle_data(
        self, query: MarketQuery, extracted: ExtractedQuery,
        tickers: list[str], now: datetime,
    ) -> MarketAnalysisResult:
        quotes, failed = self.market_data.get_quotes(tickers)
        if not quotes:
            return MarketAnalysisResult(
                intent=extracted.intent, query_tickers=tickers,
                narrative=(
                    f"I couldn't fetch data for {', '.join(tickers)} right now. "
                    "Please check the symbol(s) or try again shortly."
                ),
                disclaimer=EDUCATIONAL_DISCLAIMER, as_of=now, failed_tickers=failed,
            )

        snapshots = [build_snapshot(q) for q in quotes.values()]
        valuations = [build_valuation(q) for q in quotes.values()]
        analyst_views = [v for v in (build_analyst_view(q) for q in quotes.values()) if v]

        trends = []
        if extracted.intent in (AnalysisType.TREND, AnalysisType.COMPARE):
            for t in quotes:
                hist = self.market_data.get_history(t, query.period.value)
                if hist and hist.closes:
                    trends.append(build_trend(hist, query.period))

        as_of = min(q.as_of for q in quotes.values())
        narrative = self._narrate_data(query, extracted, snapshots, valuations, trends, analyst_views, failed)
        return MarketAnalysisResult(
            intent=extracted.intent,
            query_tickers=list(quotes.keys()),
            snapshots=snapshots,
            valuations=valuations,
            trends=trends,
            analyst_views=analyst_views,
            narrative=narrative,
            disclaimer=EDUCATIONAL_DISCLAIMER,
            as_of=as_of,
            failed_tickers=failed,
        )

    def _handle_news(
        self, query: MarketQuery, extracted: ExtractedQuery, now: datetime,
    ) -> MarketAnalysisResult:
        if not extracted.tickers:
            return self._no_ticker(query, now)
        ticker = extracted.tickers[0]

        payload = None
        if self.news_client is not None and self.news_client.is_enabled:
            payload = self.news_client.get_news_sentiment(ticker)

        if payload is None:
            return MarketAnalysisResult(
                intent=AnalysisType.NEWS, query_tickers=[ticker],
                narrative=(
                    "News sentiment isn't available right now (the news feature "
                    "needs an Alpha Vantage API key and has a small daily limit). "
                    "I can still give you a price snapshot or trend if you'd like."
                ),
                disclaimer=EDUCATIONAL_DISCLAIMER, as_of=now,
            )

        news = build_news_sentiment(ticker, payload)
        narrative = self._narrate_data(
            query, extracted, [], [], [], [], [], news=news
        )
        return MarketAnalysisResult(
            intent=AnalysisType.NEWS, query_tickers=[ticker], news=news,
            narrative=narrative, disclaimer=EDUCATIONAL_DISCLAIMER, as_of=now,
        )

    def _handle_concept(self, query: MarketQuery, now: datetime) -> MarketAnalysisResult:
        if self.retriever is None:
            return MarketAnalysisResult(
                intent=AnalysisType.CONCEPT,
                narrative=(
                    "I can look up live market data (prices, trends, metrics), but "
                    "the concept knowledge base isn't wired in here. Try asking about "
                    "a specific ticker's price or performance."
                ),
                disclaimer=EDUCATIONAL_DISCLAIMER, as_of=now,
            )

        retrieval = self.retriever.retrieve(query.question, top_k=query.top_k if hasattr(query, "top_k") else 4)
        if not retrieval.is_grounded:
            return MarketAnalysisResult(
                intent=AnalysisType.CONCEPT,
                narrative=(
                    "I don't have that market concept in my knowledge base yet. I "
                    "can help with what moves prices, reading quote data, and basic "
                    "market mechanics — or look up a specific stock for you."
                ),
                disclaimer=EDUCATIONAL_DISCLAIMER, as_of=now,
            )

        narrative = self._narrate_concept(query, retrieval.results)
        citations = self._citations_from(retrieval.results)
        return MarketAnalysisResult(
            intent=AnalysisType.CONCEPT, narrative=narrative, citations=citations,
            disclaimer=EDUCATIONAL_DISCLAIMER, as_of=now,
        )

    # --- narration ---

    def _narrate_data(self, query, extracted, snapshots, valuations, trends,
                      analyst_views, failed, news=None) -> str:
        import json

        payload = {
            "intent": extracted.intent.value,
            "snapshots": [s.model_dump() for s in snapshots],
            "valuations": [v.model_dump() for v in valuations],
            "trends": [t.model_dump(mode="json") for t in trends],
            "analyst_views": [a.model_dump() for a in analyst_views],
        }
        if news is not None:
            payload["news"] = news.model_dump(mode="json")
        if failed:
            payload["unavailable_tickers"] = failed

        user_content = f"Question: {query.question}\n\nData:\n{json.dumps(payload, default=str)}"
        try:
            resp = self.llm.invoke(
                [SystemMessage(content=_NARRATION_SYSTEM), HumanMessage(content=user_content)]
            )
            return resp.content
        except Exception:
            logger.warning("Data narration failed; using template", exc_info=True)
            return self._template_data(snapshots, trends, failed)

    def _narrate_concept(self, query, results) -> str:
        sources = "\n\n".join(
            f"[Source {i}] {sc.chunk.title} ({sc.chunk.source})\n{sc.chunk.text}"
            for i, sc in enumerate(results, start=1)
        )
        user_content = f"Question: {query.question}\n\nSources:\n{sources}"
        try:
            resp = self.llm.invoke(
                [SystemMessage(content=_CONCEPT_SYSTEM), HumanMessage(content=user_content)]
            )
            return resp.content
        except Exception:
            logger.warning("Concept narration failed; using template", exc_info=True)
            return (
                "Here's the most relevant information from my knowledge base "
                f"(AI explanation is temporarily unavailable):\n\n{results[0].chunk.text}"
            )

    @staticmethod
    def _template_data(snapshots, trends, failed) -> str:
        lines = []
        for s in snapshots:
            line = f"{s.ticker}: ${s.price:,.2f}"
            if s.change_pct is not None:
                line += f" ({s.change_pct:+.2f}% today)"
            lines.append(line)
        for t in trends:
            lines.append(
                f"{t.ticker} over {t.period.label}: {t.pct_change:+.1f}% "
                f"(trend: {t.trend_signal.value})"
            )
        if failed:
            lines.append(f"Unavailable: {', '.join(failed)}")
        lines.append("(Detailed AI explanation is temporarily unavailable.)")
        return "\n".join(lines)

    @staticmethod
    def _citations_from(results) -> list[Citation]:
        citations = []
        for sc in results:
            snippet = sc.chunk.text
            if len(snippet) > 300:
                snippet = snippet[:297].rstrip() + "..."
            citations.append(Citation(
                source=sc.chunk.source, title=sc.chunk.title, url=sc.chunk.source_url,
                snippet=snippet, relevance_score=round(sc.cosine, 4),
            ))
        return citations

    # --- helpers / guards ---

    @staticmethod
    def _is_prediction(question: str) -> bool:
        q = question.lower()
        return any(re.search(p, q) for p in _PREDICTION_PATTERNS)

    def _refusal(self, query: MarketQuery, now: datetime) -> MarketAnalysisResult:
        return MarketAnalysisResult(
            intent=AnalysisType.SNAPSHOT, narrative=_PREDICTION_REFUSAL,
            disclaimer=EDUCATIONAL_DISCLAIMER, as_of=now,
            refused=True, refusal_reason="Price predictions and personalized advice are out of scope.",
        )

    def _no_ticker(self, query: MarketQuery, now: datetime) -> MarketAnalysisResult:
        return MarketAnalysisResult(
            intent=AnalysisType.SNAPSHOT,
            narrative=(
                "I couldn't tell which stock you mean. Try naming a ticker symbol "
                "(like AAPL for Apple), and I'll pull up the data."
            ),
            disclaimer=EDUCATIONAL_DISCLAIMER, as_of=now,
            needs_confirmation=True,
            confirmation_prompt="Which ticker symbol would you like me to look up?",
        )
