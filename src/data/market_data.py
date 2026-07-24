"""
Market data service: the single place the app talks to yfinance.

Responsibilities:
- Batch quote fetching (one API round-trip for N tickers where possible)
- In-memory TTL cache (default 30 min, per course FAQ recommendation)
- Per-ticker failure isolation: one bad ticker never fails the whole request
- Asset-class mapping from yfinance quoteType

The service is deliberately synchronous and dependency-injectable so agents
can be tested with a fake fetcher and zero network access.
"""

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

import yfinance as yf

from src.agents.portfolio_analysis.model import AssetClass

logger = logging.getLogger(__name__)

DEFAULT_CACHE_TTL = timedelta(minutes=30)

# yfinance quoteType -> our coarse asset classes
_QUOTE_TYPE_MAP = {
    "EQUITY": AssetClass.EQUITY,
    "ETF": AssetClass.ETF,
    "MUTUALFUND": AssetClass.MUTUAL_FUND,
    "CRYPTOCURRENCY": AssetClass.CRYPTO,
    "MONEYMARKET": AssetClass.CASH,
}


@dataclass(frozen=True)
class Quote:
    """A point-in-time quote for one ticker."""

    ticker: str
    price: float
    currency: str
    asset_class: AssetClass
    sector: Optional[str]
    as_of: datetime


@dataclass
class QuoteResult:
    """Outcome of a batch quote request. Failures are data, not exceptions."""

    quotes: dict[str, Quote] = field(default_factory=dict)
    failed: list[str] = field(default_factory=list)


def _fetch_via_yfinance(tickers: list[str]) -> QuoteResult:
    """Real network fetch. Kept as a module-level function so tests can swap
    it for a fake without touching the cache logic."""
    result = QuoteResult()
    now = datetime.now(timezone.utc)
    for symbol in tickers:
        try:
            info = yf.Ticker(symbol).info
            price = (
                info.get("regularMarketPrice")
                or info.get("currentPrice")
                or info.get("previousClose")
            )
            if not price or price <= 0:
                logger.warning("No usable price for %s", symbol)
                result.failed.append(symbol)
                continue
            quote_type = (info.get("quoteType") or "").upper()
            result.quotes[symbol] = Quote(
                ticker=symbol,
                price=float(price),
                currency=info.get("currency") or "USD",
                asset_class=_QUOTE_TYPE_MAP.get(quote_type, AssetClass.UNKNOWN),
                sector=info.get("sector"),
                as_of=now,
            )
        except Exception:
            logger.warning("Quote fetch failed for %s", symbol, exc_info=True)
            result.failed.append(symbol)
    return result


class MarketDataService:
    """TTL-cached quote provider.

    `fetcher` is injectable: production uses yfinance, tests pass a stub.
    Thread-safe because Streamlit reruns can overlap.
    """

    def __init__(
        self,
        fetcher: Callable[[list[str]], QuoteResult] = _fetch_via_yfinance,
        cache_ttl: timedelta = DEFAULT_CACHE_TTL,
    ):
        self._fetcher = fetcher
        self._ttl = cache_ttl
        self._cache: dict[str, Quote] = {}
        self._lock = threading.Lock()

    def get_quotes(self, tickers: list[str]) -> QuoteResult:
        """Return quotes for `tickers`, serving from cache where fresh.

        Tickers that fail to fetch are listed in `result.failed`; a stale
        cached quote is preferred over a failure (better a 30-min-old price
        labeled with its timestamp than no analysis at all).
        """
        tickers = list(dict.fromkeys(t.upper() for t in tickers))  # dedupe, keep order
        now = datetime.now(timezone.utc)
        result = QuoteResult()
        to_fetch: list[str] = []

        with self._lock:
            for t in tickers:
                cached = self._cache.get(t)
                if cached and now - cached.as_of < self._ttl:
                    result.quotes[t] = cached
                else:
                    to_fetch.append(t)

        if to_fetch:
            fetched = self._fetcher(to_fetch)
            with self._lock:
                self._cache.update(fetched.quotes)
            result.quotes.update(fetched.quotes)
            for t in fetched.failed:
                stale = self._cache.get(t)
                if stale:
                    logger.info("Serving stale quote for %s after fetch failure", t)
                    result.quotes[t] = stale
                else:
                    result.failed.append(t)

        return result

    def clear_cache(self) -> None:
        with self._lock:
            self._cache.clear()
