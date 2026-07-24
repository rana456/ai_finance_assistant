"""
Portfolio Analysis Agent.

Pipeline: PortfolioInput -> market-data enrichment -> deterministic metrics
-> LLM narration -> AnalysisResult.

The LLM's only job is to explain numbers it is handed, in beginner-friendly
language. It never computes, never predicts, and never recommends specific
trades. If the LLM call fails, the agent degrades to a template narrative so
the user still gets their metrics (numbers are the product; prose is polish).
"""

import logging
from datetime import datetime, timezone

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.base import BaseFinanceAgent, EDUCATIONAL_DISCLAIMER
from src.agents.portfolio_analysis.calculator import compute_metrics, enrich_holdings
from src.agents.portfolio_analysis.model import (
    AnalysisResult,
    PortfolioInput,
    PortfolioMetrics,
)
from src.data.market_data import MarketDataService

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a financial education assistant helping beginners \
understand their investment portfolio. You will receive pre-computed portfolio \
metrics as JSON.

Rules:
- Explain what the numbers mean in plain, jargon-free language. When you use a \
financial term (diversification, concentration, asset class), define it briefly.
- ONLY discuss the numbers provided. Never invent, recompute, or adjust figures.
- Never recommend buying or selling any specific security. You may explain \
general principles (e.g. why concentration increases risk) in educational terms.
- Mention every concentration warning provided, kindly but clearly.
- Keep it under 300 words, warm and encouraging in tone.
- Do not add your own disclaimer; one is appended automatically."""


class PortfolioAnalysisAgent(BaseFinanceAgent):
    name = "portfolio_analysis"
    description = (
        "Analyzes a user's investment portfolio: current value, allocation, "
        "diversification, risk level, and gains/losses, with educational "
        "explanations."
    )

    def __init__(self, llm: BaseChatModel, market_data: MarketDataService):
        super().__init__(llm)
        self.market_data = market_data

    def run(self, portfolio: PortfolioInput) -> AnalysisResult:
        """Analyze a validated portfolio.

        Raises ValueError if no holding could be priced (nothing to analyze);
        partial quote failures are reported in `failed_tickers` instead.
        """
        tickers = [h.ticker for h in portfolio.holdings]
        quote_result = self.market_data.get_quotes(tickers)

        enriched, skipped = enrich_holdings(portfolio.holdings, quote_result.quotes)
        failed = sorted(set(quote_result.failed) | set(skipped))
        if not enriched:
            raise ValueError(
                "None of the tickers could be priced right now "
                f"({', '.join(failed)}). Please check the symbols or try again "
                "in a few minutes."
            )

        metrics = compute_metrics(enriched)
        narrative = self._narrate(metrics, failed)
        as_of = min(q.as_of for q in quote_result.quotes.values())

        return AnalysisResult(
            metrics=metrics,
            holdings=enriched,
            narrative=narrative,
            disclaimer=EDUCATIONAL_DISCLAIMER,
            failed_tickers=failed,
            as_of=as_of,
        )

    def _narrate(self, metrics: PortfolioMetrics, failed: list[str]) -> str:
        """LLM narration with template fallback on failure."""
        user_content = metrics.model_dump_json(indent=2)
        if failed:
            user_content += (
                f"\n\nNote: quotes for {', '.join(failed)} were unavailable and "
                "are excluded from the metrics. Mention this to the user."
            )
        try:
            response = self.llm.invoke(
                [SystemMessage(content=_SYSTEM_PROMPT),
                 HumanMessage(content=user_content)]
            )
            return response.content
        except Exception:
            logger.warning("LLM narration failed; using template", exc_info=True)
            return self._template_narrative(metrics, failed)

    @staticmethod
    def _template_narrative(metrics: PortfolioMetrics, failed: list[str]) -> str:
        """Deterministic fallback so the agent still delivers when the LLM is down."""
        top = metrics.allocation_by_ticker[0]
        lines = [
            f"Your portfolio is worth ${metrics.total_value:,.2f} across "
            f"{metrics.holdings_count} holding(s).",
            f"Your largest position is {top.label} at {top.percent:.1f}% of the total.",
            f"Diversification score: {metrics.diversification_score:.0f}/100. "
            f"Overall risk level: {metrics.risk_level.value}.",
        ]
        if metrics.total_gain_loss_absolute is not None:
            sign = "up" if metrics.total_gain_loss_absolute >= 0 else "down"
            lines.append(
                f"Positions with cost basis are {sign} "
                f"${abs(metrics.total_gain_loss_absolute):,.2f} "
                f"({metrics.total_gain_loss_percent:+.1f}%)."
            )
        lines.extend(metrics.concentration_warnings)
        if failed:
            lines.append(
                f"Quotes for {', '.join(failed)} were unavailable and are not "
                "included above."
            )
        lines.append("(Detailed AI explanation is temporarily unavailable.)")
        return "\n".join(lines)
