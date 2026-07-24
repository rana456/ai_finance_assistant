"""
Pydantic models for the Portfolio Analysis Agent.

Design principles (from our requirements discussion):
- One canonical internal format regardless of input source (UI form, CSV, NL chat)
- Validation happens at the model boundary, not scattered through agent code
- Ticker *format* is validated here; ticker *existence* is validated later
  against the market-data API (that's an I/O concern, not a schema concern)
"""

from datetime import date, datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class InputSource(str, Enum):
    """Where the portfolio data came from — useful for logging and for
    deciding how much confirmation to require (NL extraction needs echo-back)."""
    MANUAL = "manual"
    CSV = "csv"
    NATURAL_LANGUAGE = "natural_language"


class HoldingInput(BaseModel):
    """A single holding as entered by the user (pre-enrichment)."""

    ticker: str = Field(
        ...,
        min_length=1,
        max_length=10,
        description="Ticker symbol, e.g. 'AAPL', 'VOO'. Normalized to uppercase.",
    )
    quantity: float = Field(
        ...,
        gt=0,
        description="Number of shares. Fractional shares are allowed (e.g. 0.5).",
    )
    cost_basis_per_share: Optional[float] = Field(
        None,
        gt=0,
        description="Optional purchase price per share. Enables gain/loss calc.",
    )
    purchase_date: Optional[date] = Field(
        None,
        description="Optional purchase date.",
    )

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, v: str) -> str:
        """Uppercase, strip whitespace, and reject characters that can't
        appear in a US-listed ticker. Existence check happens later via API."""
        v = v.strip().upper()
        if not v:
            raise ValueError("Ticker cannot be empty or whitespace.")
        # Allow letters, digits, dot and hyphen (BRK.B, BF-B style tickers)
        allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-")
        if not set(v) <= allowed:
            raise ValueError(
                f"'{v}' doesn't look like a valid ticker symbol. "
                "Tickers contain only letters, numbers, '.' or '-'."
            )
        return v

    @field_validator("purchase_date")
    @classmethod
    def no_future_purchases(cls, v: Optional[date]) -> Optional[date]:
        if v is not None and v > date.today():
            raise ValueError("Purchase date cannot be in the future.")
        return v

    @field_validator("quantity")
    @classmethod
    def sane_quantity(cls, v: float) -> float:
        # Guard against absurd values that are almost certainly input errors
        # (also protects float-precision edge cases downstream).
        if v > 1e9:
            raise ValueError("Quantity looks unrealistically large — please check.")
        return v


class PortfolioInput(BaseModel):
    """The full user-submitted portfolio, before enrichment/analysis."""

    holdings: list[HoldingInput] = Field(
        ...,
        min_length=1,  # empty portfolio -> handled as a friendly agent reply,
                       # but the model itself refuses to represent one
        description="At least one holding is required for analysis.",
    )
    source: InputSource = Field(
        default=InputSource.MANUAL,
        description="How this portfolio entered the system.",
    )
    confirmed_by_user: bool = Field(
        default=False,
        description=(
            "For NATURAL_LANGUAGE input: the agent must echo the parsed "
            "holdings back and get confirmation before analysis runs."
        ),
    )

    @model_validator(mode="after")
    def merge_duplicate_tickers(self) -> "PortfolioInput":
        """User enters AAPL twice -> merge quantities instead of double-counting.
        Cost basis is merged as a weighted average when both rows have one;
        if only some rows have it, the merged holding drops it (partial data
        is worse than labeled-missing data)."""
        merged: dict[str, HoldingInput] = {}
        for h in self.holdings:
            if h.ticker not in merged:
                merged[h.ticker] = h
                continue
            existing = merged[h.ticker]
            total_qty = existing.quantity + h.quantity
            if existing.cost_basis_per_share and h.cost_basis_per_share:
                blended = (
                    existing.quantity * existing.cost_basis_per_share
                    + h.quantity * h.cost_basis_per_share
                ) / total_qty
            else:
                blended = None
            merged[h.ticker] = HoldingInput(
                ticker=h.ticker,
                quantity=total_qty,
                cost_basis_per_share=blended,
                purchase_date=None,  # ambiguous once merged
            )
        self.holdings = list(merged.values())
        return self

    @model_validator(mode="after")
    def nl_input_requires_confirmation(self) -> "PortfolioInput":
        """Enforces the echo-back rule for natural-language extraction at the
        schema level, so the agent literally cannot analyze unconfirmed
        NL-parsed portfolios."""
        if self.source == InputSource.NATURAL_LANGUAGE and not self.confirmed_by_user:
            raise ValueError(
                "Natural-language portfolios must be confirmed by the user "
                "before analysis (set confirmed_by_user=True after echo-back)."
            )
        return self


# ---------------------------------------------------------------------------
# Output models (post-enrichment, post-calculation)
# ---------------------------------------------------------------------------


class AssetClass(str, Enum):
    """Coarse asset classes used for allocation and risk assessment.
    Mapped from yfinance quoteType/sector data; UNKNOWN when the API
    can't tell us (never guessed)."""
    EQUITY = "equity"
    ETF = "etf"
    BOND = "bond"
    MUTUAL_FUND = "mutual_fund"
    CRYPTO = "crypto"
    CASH = "cash"
    UNKNOWN = "unknown"


class RiskLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


class EnrichedHolding(BaseModel):
    """A holding after market-data enrichment. All monetary values are in the
    quote currency (USD for US listings)."""

    ticker: str
    quantity: float
    current_price: float = Field(..., gt=0)
    market_value: float = Field(..., description="quantity * current_price")
    asset_class: AssetClass = AssetClass.UNKNOWN
    sector: Optional[str] = Field(
        None, description="e.g. 'Technology'. None for ETFs/funds/unknowns."
    )
    cost_basis_per_share: Optional[float] = None
    gain_loss_absolute: Optional[float] = Field(
        None, description="Only computed when cost basis was provided."
    )
    gain_loss_percent: Optional[float] = None


class AllocationSlice(BaseModel):
    """One slice of an allocation breakdown (by ticker, asset class, or sector)."""

    label: str
    value: float = Field(..., ge=0)
    percent: float = Field(..., ge=0, le=100)


class PortfolioMetrics(BaseModel):
    """Deterministic, calculator-produced numbers. The LLM narrates these;
    it never computes or alters them."""

    total_value: float = Field(..., ge=0)
    holdings_count: int = Field(..., ge=1)
    allocation_by_ticker: list[AllocationSlice]
    allocation_by_asset_class: list[AllocationSlice]
    allocation_by_sector: list[AllocationSlice] = Field(
        default_factory=list,
        description="Only includes holdings with a known sector.",
    )
    diversification_score: float = Field(
        ..., ge=0, le=100,
        description="0-100; blends holding count, concentration (HHI), and "
                    "asset-class/sector spread.",
    )
    risk_level: RiskLevel
    concentration_warnings: list[str] = Field(
        default_factory=list,
        description="Human-readable flags, e.g. single position > 30%.",
    )
    total_gain_loss_absolute: Optional[float] = Field(
        None, description="Sum over holdings that had a cost basis; None if none did."
    )
    total_gain_loss_percent: Optional[float] = None


class AnalysisResult(BaseModel):
    """Everything the agent returns to the workflow/UI layer."""

    metrics: PortfolioMetrics
    holdings: list[EnrichedHolding]
    narrative: str = Field(
        ..., description="LLM-generated educational summary of the metrics."
    )
    disclaimer: str
    failed_tickers: list[str] = Field(
        default_factory=list,
        description="Tickers we couldn't fetch a quote for; excluded from metrics.",
    )
    as_of: datetime = Field(
        ..., description="When the market data used for this analysis was fetched."
    )
