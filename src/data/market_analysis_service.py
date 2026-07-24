"""
Market analysis data service: richer yfinance access than MarketDataService.

MarketDataService serves lightweight quotes for portfolio valuation. This
service adds the two things market analysis needs — full quote/fundamental
detail and OHLCV history — behind the same design: an injectable fetcher plus
a TTL cache, so it's testable offline and gentle on the (unofficial) API.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

import yfinance as yf

logger = logging.getLogger(__name__)

QUOTE_CACHE_TTL = timedelta(minutes=30)
HISTORY_CACHE_TTL = timedelta(minutes=30)


@dataclass(frozen=True)
class DetailedQuote:
    """A rich single-ticker snapshot straight from the data source."""

    ticker: str
    price: float
    as_of: datetime
    long_name: Optional[str] = None
    previous_close: Optional[float] = None
    day_low: Optional[float] = None
    day_high: Optional[float] = None
    volume: Optional[int] = None
    avg_volume: Optional[int] = None
    week52_low: Optional[float] = None
    week52_high: Optional[float] = None
    market_cap: Optional[int] = None
    currency: str = "USD"
    trailing_pe: Optional[float] = None
    forward_pe: Optional[float] = None
    price_to_book: Optional[float] = None
    peg_ratio: Optional[float] = None
    eps: Optional[float] = None
    dividend_yield: Optional[float] = None
    beta: Optional[float] = None
    recommendation: Optional[str] = None
    target_low: Optional[float] = None
    target_mean: Optional[float] = None
    target_high: Optional[float] = None
    num_analysts: Optional[int] = None


@dataclass(frozen=True)
class PriceHistory:
    """Daily closing prices for a ticker over some period."""

    ticker: str
    period: str
    closes: list[float]
    as_of: datetime


@dataclass
class _Cached:
    value: object
    stored_at: datetime


def _fetch_detailed_quote(ticker: str) -> Optional[DetailedQuote]:
    """Real yfinance fetch for one ticker's detail. Returns None on failure."""
    try:
        info = yf.Ticker(ticker).info
        price = (
            info.get("currentPrice")
            or info.get("regularMarketPrice")
            or info.get("previousClose")
        )
        if not price or price <= 0:
            return None
        return DetailedQuote(
            ticker=ticker,
            price=float(price),
            as_of=datetime.now(timezone.utc),
            long_name=info.get("longName"),
            previous_close=info.get("previousClose"),
            day_low=info.get("dayLow"),
            day_high=info.get("dayHigh"),
            volume=info.get("volume") or info.get("regularMarketVolume"),
            avg_volume=info.get("averageVolume"),
            week52_low=info.get("fiftyTwoWeekLow"),
            week52_high=info.get("fiftyTwoWeekHigh"),
            market_cap=info.get("marketCap"),
            currency=info.get("currency") or "USD",
            trailing_pe=info.get("trailingPE"),
            forward_pe=info.get("forwardPE"),
            price_to_book=info.get("priceToBook"),
            peg_ratio=info.get("pegRatio"),
            eps=info.get("trailingEps"),
            dividend_yield=info.get("dividendYield"),
            beta=info.get("beta"),
            recommendation=info.get("recommendationKey"),
            target_low=info.get("targetLowPrice"),
            target_mean=info.get("targetMeanPrice"),
            target_high=info.get("targetHighPrice"),
            num_analysts=info.get("numberOfAnalystOpinions"),
        )
    except Exception:
        logger.warning("Detailed quote fetch failed for %s", ticker, exc_info=True)
        return None


def _fetch_history(ticker: str, period: str) -> Optional[PriceHistory]:
    """Real yfinance history fetch. Returns None on failure/empty."""
    try:
        df = yf.Ticker(ticker).history(period=period)
        if df is None or df.empty or "Close" not in df:
            return None
        closes = [float(c) for c in df["Close"].tolist() if c == c]  # drop NaN
        if not closes:
            return None
        return PriceHistory(
            ticker=ticker, period=period, closes=closes,
            as_of=datetime.now(timezone.utc),
        )
    except Exception:
        logger.warning("History fetch failed for %s", ticker, exc_info=True)
        return None


class MarketAnalysisService:
    """Cached, injectable provider of detailed quotes and price history."""

    def __init__(
        self,
        quote_fetcher: Callable[[str], Optional[DetailedQuote]] = _fetch_detailed_quote,
        history_fetcher: Callable[[str, str], Optional[PriceHistory]] = _fetch_history,
        quote_ttl: timedelta = QUOTE_CACHE_TTL,
        history_ttl: timedelta = HISTORY_CACHE_TTL,
    ):
        self._quote_fetcher = quote_fetcher
        self._history_fetcher = history_fetcher
        self._quote_ttl = quote_ttl
        self._history_ttl = history_ttl
        self._quote_cache: dict[str, _Cached] = {}
        self._history_cache: dict[tuple[str, str], _Cached] = {}
        self._lock = threading.Lock()

    def get_quote(self, ticker: str) -> Optional[DetailedQuote]:
        ticker = ticker.upper()
        now = datetime.now(timezone.utc)
        with self._lock:
            hit = self._quote_cache.get(ticker)
            if hit and now - hit.stored_at < self._quote_ttl:
                return hit.value  # type: ignore[return-value]
        quote = self._quote_fetcher(ticker)
        if quote is not None:
            with self._lock:
                self._quote_cache[ticker] = _Cached(quote, now)
        return quote

    def get_quotes(self, tickers: list[str]) -> tuple[dict[str, DetailedQuote], list[str]]:
        """Batch helper: returns (quotes_by_ticker, failed_tickers)."""
        quotes: dict[str, DetailedQuote] = {}
        failed: list[str] = []
        for t in dict.fromkeys(t.upper() for t in tickers):
            q = self.get_quote(t)
            if q is None:
                failed.append(t)
            else:
                quotes[t] = q
        return quotes, failed

    def get_history(self, ticker: str, period: str) -> Optional[PriceHistory]:
        ticker = ticker.upper()
        key = (ticker, period)
        now = datetime.now(timezone.utc)
        with self._lock:
            hit = self._history_cache.get(key)
            if hit and now - hit.stored_at < self._history_ttl:
                return hit.value  # type: ignore[return-value]
        history = self._history_fetcher(ticker, period)
        if history is not None:
            with self._lock:
                self._history_cache[key] = _Cached(history, now)
        return history

    def clear_cache(self) -> None:
        with self._lock:
            self._quote_cache.clear()
            self._history_cache.clear()
