"""
Pure calculation functions for market analysis.

Deterministic and side-effect free: they turn raw service data (DetailedQuote,
PriceHistory, news payloads) into the typed numeric models. The LLM narrates
these; it never computes them.
"""

from __future__ import annotations

from typing import Optional

from src.agents.market_analysis.model import (
    AnalystView,
    NewsArticle,
    NewsSentiment,
    Period,
    QuoteSnapshot,
    SentimentLabel,
    TrendAnalysis,
    TrendSignal,
    ValuationMetrics,
)
from src.data.market_analysis_service import DetailedQuote, PriceHistory

# A period move beyond this (in %) reads as a real trend when moving averages
# aren't available (short look-backs).
_TREND_PCT_THRESHOLD = 3.0


def build_snapshot(q: DetailedQuote) -> QuoteSnapshot:
    """Derive change, volume-vs-average, and 52-week positioning from a quote."""
    change_abs = change_pct = None
    if q.previous_close:
        change_abs = q.price - q.previous_close
        change_pct = (change_abs / q.previous_close) * 100

    volume_vs_avg = None
    if q.volume and q.avg_volume:
        volume_vs_avg = (q.volume / q.avg_volume) * 100

    pct_in_range = None
    if q.week52_low is not None and q.week52_high is not None:
        span = q.week52_high - q.week52_low
        if span > 0:
            pct_in_range = max(0.0, min(100.0, ((q.price - q.week52_low) / span) * 100))

    return QuoteSnapshot(
        ticker=q.ticker,
        long_name=q.long_name,
        price=q.price,
        previous_close=q.previous_close,
        change_abs=change_abs,
        change_pct=change_pct,
        day_low=q.day_low,
        day_high=q.day_high,
        volume=q.volume,
        avg_volume=q.avg_volume,
        volume_vs_avg_pct=volume_vs_avg,
        week52_low=q.week52_low,
        week52_high=q.week52_high,
        pct_in_52wk_range=pct_in_range,
        market_cap=q.market_cap,
        currency=q.currency,
    )


def build_valuation(q: DetailedQuote) -> ValuationMetrics:
    """Pass-through of valuation ratios (already numeric from the source)."""
    return ValuationMetrics(
        ticker=q.ticker,
        trailing_pe=q.trailing_pe,
        forward_pe=q.forward_pe,
        price_to_book=q.price_to_book,
        peg_ratio=q.peg_ratio,
        eps=q.eps,
        dividend_yield=q.dividend_yield,
        beta=q.beta,
    )


def build_analyst_view(q: DetailedQuote) -> Optional[AnalystView]:
    """Analyst opinion, or None if the source reports none (e.g. most ETFs)."""
    if not any([q.recommendation, q.target_mean, q.num_analysts]):
        return None
    return AnalystView(
        ticker=q.ticker,
        recommendation=q.recommendation,
        target_low=q.target_low,
        target_mean=q.target_mean,
        target_high=q.target_high,
        num_analysts=q.num_analysts,
    )


def _sma(closes: list[float], window: int) -> Optional[float]:
    """Simple moving average of the last `window` closes, or None if too few."""
    if len(closes) < window:
        return None
    return sum(closes[-window:]) / window


def _trend_signal(
    pct_change: float, sma_50: Optional[float], sma_200: Optional[float], last: float
) -> TrendSignal:
    """Prefer moving-average structure when available; otherwise fall back to
    the size of the period move."""
    if sma_50 is not None and sma_200 is not None:
        # Golden/death-cross style read, confirmed by price vs. the long average.
        if sma_50 > sma_200 and last >= sma_200:
            return TrendSignal.UPTREND
        if sma_50 < sma_200 and last <= sma_200:
            return TrendSignal.DOWNTREND
        return TrendSignal.SIDEWAYS
    if pct_change > _TREND_PCT_THRESHOLD:
        return TrendSignal.UPTREND
    if pct_change < -_TREND_PCT_THRESHOLD:
        return TrendSignal.DOWNTREND
    return TrendSignal.SIDEWAYS


def build_trend(history: PriceHistory, period: Period) -> TrendAnalysis:
    """Compute period return, range, moving averages, and a trend signal.

    Raises ValueError on empty history (agent turns that into a friendly note).
    """
    closes = history.closes
    if not closes:
        raise ValueError(f"No price history available for {history.ticker}.")

    start, end = closes[0], closes[-1]
    pct_change = ((end - start) / start) * 100 if start else 0.0
    sma_50 = _sma(closes, 50)
    sma_200 = _sma(closes, 200)

    return TrendAnalysis(
        ticker=history.ticker,
        period=period,
        start_price=start,
        end_price=end,
        pct_change=pct_change,
        period_low=min(closes),
        period_high=max(closes),
        sma_50=sma_50,
        sma_200=sma_200,
        trend_signal=_trend_signal(pct_change, sma_50, sma_200, end),
    )


_LABEL_MAP = {
    "bullish": SentimentLabel.BULLISH,
    "somewhat-bullish": SentimentLabel.SOMEWHAT_BULLISH,
    "neutral": SentimentLabel.NEUTRAL,
    "somewhat-bearish": SentimentLabel.SOMEWHAT_BEARISH,
    "bearish": SentimentLabel.BEARISH,
}


def _map_label(raw: Optional[str]) -> Optional[SentimentLabel]:
    if not raw:
        return None
    return _LABEL_MAP.get(raw.strip().lower().replace("_", "-"))


def build_news_sentiment(ticker: str, payload: dict) -> NewsSentiment:
    """Parse an Alpha Vantage NEWS_SENTIMENT payload into our model, keeping
    only this ticker's per-article sentiment and averaging it for an overall."""
    articles: list[NewsArticle] = []
    scores: list[float] = []
    for item in (payload.get("feed") or [])[:5]:
        ticker_score = None
        ticker_label = None
        for ts in item.get("ticker_sentiment", []):
            if ts.get("ticker", "").upper() == ticker.upper():
                try:
                    ticker_score = float(ts.get("ticker_sentiment_score"))
                except (TypeError, ValueError):
                    ticker_score = None
                ticker_label = _map_label(ts.get("ticker_sentiment_label"))
                break
        if ticker_score is not None:
            scores.append(ticker_score)
        articles.append(
            NewsArticle(
                title=item.get("title", "(untitled)"),
                url=item.get("url"),
                source=item.get("source"),
                summary=item.get("summary"),
                sentiment_label=ticker_label,
                sentiment_score=ticker_score,
                time_published=item.get("time_published"),
            )
        )

    overall_score = sum(scores) / len(scores) if scores else None
    overall_label = None
    if overall_score is not None:
        # Alpha Vantage's documented score-to-label thresholds.
        if overall_score <= -0.35:
            overall_label = SentimentLabel.BEARISH
        elif overall_score <= -0.15:
            overall_label = SentimentLabel.SOMEWHAT_BEARISH
        elif overall_score < 0.15:
            overall_label = SentimentLabel.NEUTRAL
        elif overall_score < 0.35:
            overall_label = SentimentLabel.SOMEWHAT_BULLISH
        else:
            overall_label = SentimentLabel.BULLISH

    return NewsSentiment(
        ticker=ticker,
        articles=articles,
        overall_sentiment_score=overall_score,
        overall_label=overall_label,
    )
