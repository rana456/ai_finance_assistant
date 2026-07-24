"""
Pydantic models for the Market Analysis Agent.

Same discipline as the other agents: the deterministic layer fills these
numeric models; the LLM only narrates them. Every result carries an `as_of`
timestamp (the data-freshness indicator required by the milestones) and a
disclaimer.
"""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator

from src.agents.finance_qa.model import Citation, KnowledgeLevel


class AnalysisType(str, Enum):
    """What the user is asking for. Drives routing and which output sections
    get populated."""
    SNAPSHOT = "snapshot"     # current quote for a ticker
    METRIC = "metric"         # a specific valuation metric, explained
    TREND = "trend"           # performance over a period
    COMPARE = "compare"       # side-by-side of several tickers
    OVERVIEW = "overview"     # major market indices
    NEWS = "news"             # news + sentiment for a ticker
    CONCEPT = "concept"       # "what moves prices" -> routed to RAG


class Period(str, Enum):
    """Look-back window for trend analysis. Value is the yfinance period arg."""
    ONE_DAY = "1d"
    FIVE_DAY = "5d"
    ONE_MONTH = "1mo"
    SIX_MONTH = "6mo"
    ONE_YEAR = "1y"
    FIVE_YEAR = "5y"

    @property
    def label(self) -> str:
        return {
            "1d": "1 day", "5d": "5 days", "1mo": "1 month",
            "6mo": "6 months", "1y": "1 year", "5y": "5 years",
        }[self.value]


class TrendSignal(str, Enum):
    UPTREND = "uptrend"
    DOWNTREND = "downtrend"
    SIDEWAYS = "sideways"


class SentimentLabel(str, Enum):
    """Alpha Vantage's sentiment buckets."""
    BULLISH = "bullish"
    SOMEWHAT_BULLISH = "somewhat_bullish"
    NEUTRAL = "neutral"
    SOMEWHAT_BEARISH = "somewhat_bearish"
    BEARISH = "bearish"


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------


class MarketQuery(BaseModel):
    """A question routed to the Market Analysis agent."""

    question: str = Field(..., min_length=1, description="Natural-language question.")
    tickers: Optional[list[str]] = Field(
        None, description="Explicit tickers; if None they're extracted from the question."
    )
    analysis_type: Optional[AnalysisType] = Field(
        None, description="Explicit intent; if None it's inferred from the question."
    )
    period: Period = Field(
        default=Period.ONE_MONTH, description="Look-back window for trend analysis."
    )
    knowledge_level: KnowledgeLevel = KnowledgeLevel.BEGINNER

    @field_validator("question")
    @classmethod
    def non_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Question cannot be empty.")
        return v

    @field_validator("tickers")
    @classmethod
    def normalize_tickers(cls, v: Optional[list[str]]) -> Optional[list[str]]:
        if v is None:
            return None
        return [t.strip().upper() for t in v if t.strip()]


# ---------------------------------------------------------------------------
# Output sections
# ---------------------------------------------------------------------------


class QuoteSnapshot(BaseModel):
    """Current-quote view of one ticker, with derived positioning."""

    ticker: str
    long_name: Optional[str] = None
    price: float
    previous_close: Optional[float] = None
    change_abs: Optional[float] = Field(None, description="price - previous_close")
    change_pct: Optional[float] = None
    day_low: Optional[float] = None
    day_high: Optional[float] = None
    volume: Optional[int] = None
    avg_volume: Optional[int] = None
    volume_vs_avg_pct: Optional[float] = Field(
        None, description="Today's volume vs. average, as a percentage."
    )
    week52_low: Optional[float] = None
    week52_high: Optional[float] = None
    pct_in_52wk_range: Optional[float] = Field(
        None, ge=0, le=100,
        description="0 = at 52-wk low, 100 = at 52-wk high.",
    )
    market_cap: Optional[int] = None
    currency: str = "USD"


class ValuationMetrics(BaseModel):
    """Valuation and fundamental ratios. All optional — not every ticker
    (e.g. an ETF) exposes every field."""

    ticker: str
    trailing_pe: Optional[float] = None
    forward_pe: Optional[float] = None
    price_to_book: Optional[float] = None
    peg_ratio: Optional[float] = None
    eps: Optional[float] = None
    dividend_yield: Optional[float] = None
    beta: Optional[float] = None


class TrendAnalysis(BaseModel):
    """Performance over a look-back period, plus moving-average context."""

    ticker: str
    period: Period
    start_price: float
    end_price: float
    pct_change: float
    period_low: float
    period_high: float
    sma_50: Optional[float] = Field(None, description="50-day simple moving average.")
    sma_200: Optional[float] = Field(None, description="200-day simple moving average.")
    trend_signal: TrendSignal


class AnalystView(BaseModel):
    """Aggregated sell-side analyst opinion (as reported by the data source)."""

    ticker: str
    recommendation: Optional[str] = None
    target_low: Optional[float] = None
    target_mean: Optional[float] = None
    target_high: Optional[float] = None
    num_analysts: Optional[int] = None


class NewsArticle(BaseModel):
    title: str
    url: Optional[str] = None
    source: Optional[str] = None
    summary: Optional[str] = None
    sentiment_label: Optional[SentimentLabel] = None
    sentiment_score: Optional[float] = None
    time_published: Optional[str] = None


class NewsSentiment(BaseModel):
    """News + sentiment for a ticker (Alpha Vantage). Optional feature."""

    ticker: str
    articles: list[NewsArticle] = Field(default_factory=list)
    overall_sentiment_score: Optional[float] = None
    overall_label: Optional[SentimentLabel] = None


# ---------------------------------------------------------------------------
# Top-level result
# ---------------------------------------------------------------------------


class MarketAnalysisResult(BaseModel):
    """Everything the agent returns. Only the sections relevant to the detected
    intent are populated."""

    intent: AnalysisType
    query_tickers: list[str] = Field(default_factory=list)
    snapshots: list[QuoteSnapshot] = Field(default_factory=list)
    valuations: list[ValuationMetrics] = Field(default_factory=list)
    trends: list[TrendAnalysis] = Field(default_factory=list)
    analyst_views: list[AnalystView] = Field(default_factory=list)
    news: Optional[NewsSentiment] = None
    narrative: str
    citations: list[Citation] = Field(default_factory=list)
    disclaimer: str
    as_of: datetime = Field(..., description="When the underlying data was fetched.")
    failed_tickers: list[str] = Field(default_factory=list)
    refused: bool = False
    refusal_reason: Optional[str] = None
    needs_confirmation: bool = Field(
        default=False, description="True when extracted tickers need user echo-back."
    )
    confirmation_prompt: Optional[str] = None
