"""
Pure calculation functions for portfolio analysis.

Everything here is deterministic and side-effect free: no network, no LLM,
no clock reads. The agent feeds these numbers to the LLM for narration;
the LLM never computes or modifies them.

Metric definitions (per course FAQ: total value, allocation percentages,
basic diversification score, simple risk assessment by asset type):

- Diversification score (0-100): blend of
    * holding-count factor (more positions -> better, saturates at 15)
    * concentration factor (1 - HHI over ticker allocation)
    * asset-class spread factor (1 - HHI over asset-class allocation)
- Risk level: driven by asset-class weights (crypto/single stocks risky,
  bonds/cash safe) plus a concentration penalty.
"""

from src.agents.portfolio_analysis.model import (
    AllocationSlice,
    AssetClass,
    EnrichedHolding,
    HoldingInput,
    PortfolioMetrics,
    RiskLevel,
)
from src.data.market_data import Quote

# Single position above this share of the portfolio triggers a warning.
CONCENTRATION_THRESHOLD = 0.30

# Risk weight per asset class, 0 (safest) .. 1 (riskiest).
_ASSET_RISK_WEIGHT = {
    AssetClass.CASH: 0.0,
    AssetClass.BOND: 0.2,
    AssetClass.MUTUAL_FUND: 0.4,
    AssetClass.ETF: 0.45,
    AssetClass.EQUITY: 0.7,
    AssetClass.UNKNOWN: 0.7,  # unknown is treated as risky, not ignored
    AssetClass.CRYPTO: 1.0,
}


def enrich_holdings(
    holdings: list[HoldingInput], quotes: dict[str, Quote]
) -> tuple[list[EnrichedHolding], list[str]]:
    """Join user holdings with market quotes.

    Returns (enriched, skipped_tickers). Holdings without a quote are skipped
    and reported — partial analysis with labeled gaps beats no analysis.
    """
    enriched: list[EnrichedHolding] = []
    skipped: list[str] = []
    for h in holdings:
        quote = quotes.get(h.ticker)
        if quote is None:
            skipped.append(h.ticker)
            continue
        market_value = h.quantity * quote.price
        gain_abs = gain_pct = None
        if h.cost_basis_per_share:
            cost_total = h.quantity * h.cost_basis_per_share
            gain_abs = market_value - cost_total
            gain_pct = (gain_abs / cost_total) * 100
        enriched.append(
            EnrichedHolding(
                ticker=h.ticker,
                quantity=h.quantity,
                current_price=quote.price,
                market_value=market_value,
                asset_class=quote.asset_class,
                sector=quote.sector,
                cost_basis_per_share=h.cost_basis_per_share,
                gain_loss_absolute=gain_abs,
                gain_loss_percent=gain_pct,
            )
        )
    return enriched, skipped


def _allocation(values: dict[str, float], total: float) -> list[AllocationSlice]:
    """Build percentage slices sorted largest-first."""
    slices = [
        AllocationSlice(label=label, value=value, percent=(value / total) * 100)
        for label, value in values.items()
    ]
    return sorted(slices, key=lambda s: s.value, reverse=True)


def _hhi(slices: list[AllocationSlice]) -> float:
    """Herfindahl–Hirschman index over allocation fractions: 1/N (perfectly
    even, N slices) .. 1.0 (everything in one slice)."""
    return sum((s.percent / 100) ** 2 for s in slices)


def compute_metrics(enriched: list[EnrichedHolding]) -> PortfolioMetrics:
    """Compute all portfolio metrics from enriched holdings.

    Raises ValueError on an empty list — the agent layer is responsible for
    turning that into a friendly message.
    """
    if not enriched:
        raise ValueError("Cannot compute metrics for an empty portfolio.")

    total = sum(h.market_value for h in enriched)

    by_ticker: dict[str, float] = {}
    by_class: dict[str, float] = {}
    by_sector: dict[str, float] = {}
    for h in enriched:
        by_ticker[h.ticker] = by_ticker.get(h.ticker, 0) + h.market_value
        by_class[h.asset_class.value] = (
            by_class.get(h.asset_class.value, 0) + h.market_value
        )
        if h.sector:
            by_sector[h.sector] = by_sector.get(h.sector, 0) + h.market_value

    ticker_alloc = _allocation(by_ticker, total)
    class_alloc = _allocation(by_class, total)
    sector_total = sum(by_sector.values())
    sector_alloc = _allocation(by_sector, sector_total) if by_sector else []

    # --- Diversification score ---
    count_factor = min(len(enriched), 15) / 15
    concentration_factor = 1 - _hhi(ticker_alloc)
    class_factor = 1 - _hhi(class_alloc)
    score = 100 * (
        0.35 * count_factor + 0.45 * concentration_factor + 0.20 * class_factor
    )

    # --- Concentration warnings ---
    warnings = [
        f"{s.label} is {s.percent:.1f}% of your portfolio — above the "
        f"{CONCENTRATION_THRESHOLD:.0%} concentration guideline."
        for s in ticker_alloc
        if s.percent / 100 > CONCENTRATION_THRESHOLD
    ]

    # --- Risk level ---
    weighted_risk = sum(
        _ASSET_RISK_WEIGHT[h.asset_class] * h.market_value for h in enriched
    ) / total
    if warnings:  # concentration pushes risk up a notch
        weighted_risk = min(1.0, weighted_risk + 0.15)
    if weighted_risk < 0.35:
        risk = RiskLevel.LOW
    elif weighted_risk < 0.65:
        risk = RiskLevel.MODERATE
    else:
        risk = RiskLevel.HIGH

    # --- Aggregate gain/loss (only over holdings that had a cost basis) ---
    with_basis = [h for h in enriched if h.gain_loss_absolute is not None]
    total_gain_abs = total_gain_pct = None
    if with_basis:
        total_gain_abs = sum(h.gain_loss_absolute for h in with_basis)
        total_cost = sum(h.quantity * h.cost_basis_per_share for h in with_basis)
        total_gain_pct = (total_gain_abs / total_cost) * 100

    return PortfolioMetrics(
        total_value=total,
        holdings_count=len(enriched),
        allocation_by_ticker=ticker_alloc,
        allocation_by_asset_class=class_alloc,
        allocation_by_sector=sector_alloc,
        diversification_score=round(score, 1),
        risk_level=risk,
        concentration_warnings=warnings,
        total_gain_loss_absolute=total_gain_abs,
        total_gain_loss_percent=total_gain_pct,
    )
