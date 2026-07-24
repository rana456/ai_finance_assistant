"""Shared fixtures: fake quotes, fake market data fetcher, fake embedder."""

import hashlib
import re
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from src.agents.portfolio_analysis.model import AssetClass
from src.data.market_data import MarketDataService, Quote, QuoteResult

NOW = datetime(2026, 7, 20, 12, 0, tzinfo=timezone.utc)


def make_quote(
    ticker: str,
    price: float,
    asset_class: AssetClass = AssetClass.EQUITY,
    sector: str | None = None,
    as_of: datetime = NOW,
) -> Quote:
    return Quote(
        ticker=ticker,
        price=price,
        currency="USD",
        asset_class=asset_class,
        sector=sector,
        as_of=as_of,
    )


SAMPLE_QUOTES = {
    "AAPL": make_quote("AAPL", 200.0, AssetClass.EQUITY, "Technology"),
    "MSFT": make_quote("MSFT", 400.0, AssetClass.EQUITY, "Technology"),
    "VOO": make_quote("VOO", 500.0, AssetClass.ETF, None),
    "BND": make_quote("BND", 75.0, AssetClass.ETF, None),
    "JNJ": make_quote("JNJ", 150.0, AssetClass.EQUITY, "Healthcare"),
}


class FakeFetcher:
    """Injectable fetcher that serves canned quotes and counts calls."""

    def __init__(self, quotes: dict[str, Quote] | None = None):
        self.quotes = dict(quotes or SAMPLE_QUOTES)
        self.calls: list[list[str]] = []

    def __call__(self, tickers: list[str]) -> QuoteResult:
        self.calls.append(list(tickers))
        result = QuoteResult()
        fetched_at = datetime.now(timezone.utc)  # fresh stamp, like the real fetcher
        for t in tickers:
            if t in self.quotes:
                result.quotes[t] = replace(self.quotes[t], as_of=fetched_at)
            else:
                result.failed.append(t)
        return result


@pytest.fixture
def fake_fetcher() -> FakeFetcher:
    return FakeFetcher()


@pytest.fixture
def market_data(fake_fetcher: FakeFetcher) -> MarketDataService:
    return MarketDataService(fetcher=fake_fetcher)


# ---------------------------------------------------------------------------
# Finance Q&A / RAG fixtures
# ---------------------------------------------------------------------------


class FakeEmbedder:
    """Deterministic, network-free embedder for tests.

    Uses the hashing trick: each token increments a bucket in a fixed-width
    vector. Texts that share tokens get overlapping vectors and thus positive
    cosine similarity, so nearest-neighbor behavior is realistic enough to
    exercise the dense-retrieval plumbing — without OpenAI or a key.
    """

    dimension = 128

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dimension
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            bucket = int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dimension
            vec[bucket] += 1.0
        return vec

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


# ---------------------------------------------------------------------------
# Market Analysis fixtures
# ---------------------------------------------------------------------------

from src.data.market_analysis_service import (  # noqa: E402
    DetailedQuote, MarketAnalysisService, PriceHistory,
)

MARKET_NOW = datetime(2026, 7, 21, 16, 0, tzinfo=timezone.utc)


def make_detailed(ticker: str, price: float, **kw) -> DetailedQuote:
    base = dict(
        previous_close=price * 0.99, day_low=price * 0.98, day_high=price * 1.02,
        volume=40_000_000, avg_volume=55_000_000,
        week52_low=price * 0.6, week52_high=price * 1.1,
        market_cap=int(price * 1e9), currency="USD",
        trailing_pe=30.0, forward_pe=25.0, price_to_book=10.0, peg_ratio=2.0,
        eps=price / 30, dividend_yield=0.5, beta=1.1,
        recommendation="buy", target_low=price * 0.8, target_mean=price * 1.1,
        target_high=price * 1.4, num_analysts=40, long_name=f"{ticker} Inc.",
        as_of=MARKET_NOW,
    )
    base.update(kw)
    return DetailedQuote(ticker=ticker, price=price, **base)


def _linear(start: float, end: float, n: int = 250) -> list[float]:
    return [start + (end - start) * i / (n - 1) for i in range(n)]


SAMPLE_DETAILED = {
    "AAPL": make_detailed("AAPL", 200.0),
    "MSFT": make_detailed("MSFT", 400.0),
    # ETF: no analyst coverage, thin fundamentals
    "VOO": make_detailed("VOO", 500.0, recommendation=None, target_mean=None,
                         num_analysts=None, trailing_pe=None, eps=None),
    "^GSPC": make_detailed("^GSPC", 5500.0, recommendation=None, target_mean=None,
                          num_analysts=None, trailing_pe=None, eps=None),
    "^IXIC": make_detailed("^IXIC", 18000.0, recommendation=None, target_mean=None,
                          num_analysts=None, trailing_pe=None, eps=None),
    "^DJI": make_detailed("^DJI", 42000.0, recommendation=None, target_mean=None,
                         num_analysts=None, trailing_pe=None, eps=None),
}

SAMPLE_HISTORY = {
    ("AAPL", "1mo"): _linear(180.0, 200.0),   # uptrend
    ("AAPL", "1y"): _linear(150.0, 200.0),    # uptrend, enough for SMA-200
    ("MSFT", "1y"): _linear(450.0, 400.0),    # downtrend
    ("VOO", "1y"): _linear(500.0, 500.0),     # flat/sideways
}


class FakeQuoteFetcher:
    def __init__(self, quotes=None):
        self.quotes = dict(quotes or SAMPLE_DETAILED)
        self.calls: list[str] = []

    def __call__(self, ticker: str):
        self.calls.append(ticker)
        q = self.quotes.get(ticker.upper())
        return replace(q, as_of=datetime.now(timezone.utc)) if q else None


class FakeHistoryFetcher:
    def __init__(self, histories=None):
        self.histories = dict(histories or SAMPLE_HISTORY)
        self.calls: list[tuple[str, str]] = []

    def __call__(self, ticker: str, period: str):
        self.calls.append((ticker, period))
        closes = self.histories.get((ticker.upper(), period))
        if closes is None:
            return None
        return PriceHistory(ticker=ticker.upper(), period=period,
                            closes=list(closes), as_of=datetime.now(timezone.utc))


@pytest.fixture
def fake_quote_fetcher() -> FakeQuoteFetcher:
    return FakeQuoteFetcher()


@pytest.fixture
def fake_history_fetcher() -> FakeHistoryFetcher:
    return FakeHistoryFetcher()


@pytest.fixture
def market_analysis_service(fake_quote_fetcher, fake_history_fetcher) -> MarketAnalysisService:
    return MarketAnalysisService(
        quote_fetcher=fake_quote_fetcher, history_fetcher=fake_history_fetcher
    )
