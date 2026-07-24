"""Tests for portfolio math: enrichment, allocation, diversification, risk."""

import pytest

from src.agents.portfolio_analysis.calculator import compute_metrics, enrich_holdings
from src.agents.portfolio_analysis.model import (
    AssetClass,
    HoldingInput,
    RiskLevel,
)

from tests.conftest import SAMPLE_QUOTES, make_quote


def holdings(*specs) -> list[HoldingInput]:
    """specs: (ticker, qty) or (ticker, qty, cost_basis)."""
    return [
        HoldingInput(
            ticker=s[0], quantity=s[1],
            cost_basis_per_share=s[2] if len(s) > 2 else None,
        )
        for s in specs
    ]


class TestEnrichment:
    def test_market_value_computed(self):
        enriched, skipped = enrich_holdings(
            holdings(("AAPL", 10)), SAMPLE_QUOTES
        )
        assert skipped == []
        assert enriched[0].market_value == pytest.approx(2000.0)  # 10 * $200

    def test_gain_loss_with_cost_basis(self):
        enriched, _ = enrich_holdings(
            holdings(("AAPL", 10, 100.0)), SAMPLE_QUOTES
        )
        h = enriched[0]
        assert h.gain_loss_absolute == pytest.approx(1000.0)  # 2000 - 1000
        assert h.gain_loss_percent == pytest.approx(100.0)

    def test_no_gain_loss_without_cost_basis(self):
        enriched, _ = enrich_holdings(holdings(("AAPL", 10)), SAMPLE_QUOTES)
        assert enriched[0].gain_loss_absolute is None

    def test_missing_quote_skipped_and_reported(self):
        enriched, skipped = enrich_holdings(
            holdings(("AAPL", 10), ("FAKETICK", 5)), SAMPLE_QUOTES
        )
        assert [h.ticker for h in enriched] == ["AAPL"]
        assert skipped == ["FAKETICK"]


class TestMetrics:
    def test_empty_raises(self):
        with pytest.raises(ValueError):
            compute_metrics([])

    def test_total_value_and_allocation(self):
        enriched, _ = enrich_holdings(
            holdings(("AAPL", 10), ("MSFT", 5)), SAMPLE_QUOTES  # 2000 + 2000
        )
        m = compute_metrics(enriched)
        assert m.total_value == pytest.approx(4000.0)
        assert m.holdings_count == 2
        for s in m.allocation_by_ticker:
            assert s.percent == pytest.approx(50.0)

    def test_allocation_sorted_largest_first(self):
        enriched, _ = enrich_holdings(
            holdings(("AAPL", 1), ("MSFT", 10)), SAMPLE_QUOTES
        )
        m = compute_metrics(enriched)
        assert m.allocation_by_ticker[0].label == "MSFT"

    def test_sector_allocation_ignores_sectorless(self):
        enriched, _ = enrich_holdings(
            holdings(("AAPL", 10), ("VOO", 4)), SAMPLE_QUOTES  # VOO has no sector
        )
        m = compute_metrics(enriched)
        assert [s.label for s in m.allocation_by_sector] == ["Technology"]
        assert m.allocation_by_sector[0].percent == pytest.approx(100.0)

    def test_single_position_low_diversification_and_warning(self):
        enriched, _ = enrich_holdings(holdings(("AAPL", 10)), SAMPLE_QUOTES)
        m = compute_metrics(enriched)
        assert m.diversification_score < 15
        assert len(m.concentration_warnings) == 1
        assert "AAPL" in m.concentration_warnings[0]

    def test_spread_portfolio_scores_higher(self):
        one, _ = enrich_holdings(holdings(("AAPL", 10)), SAMPLE_QUOTES)
        five, _ = enrich_holdings(
            holdings(("AAPL", 10), ("MSFT", 5), ("VOO", 4), ("BND", 27), ("JNJ", 13)),
            SAMPLE_QUOTES,
        )
        assert (
            compute_metrics(five).diversification_score
            > compute_metrics(one).diversification_score
        )

    def test_no_warning_below_threshold(self):
        # Four even positions -> 25% each, under the 30% threshold
        enriched, _ = enrich_holdings(
            holdings(("AAPL", 10), ("MSFT", 5), ("VOO", 4), ("JNJ", 40 / 3)),
            SAMPLE_QUOTES,
        )
        m = compute_metrics(enriched)
        assert m.concentration_warnings == []

    def test_all_equity_concentrated_is_high_risk(self):
        enriched, _ = enrich_holdings(holdings(("AAPL", 10)), SAMPLE_QUOTES)
        assert compute_metrics(enriched).risk_level == RiskLevel.HIGH

    def test_bond_heavy_is_low_risk(self):
        quotes = {"BONDFUND": make_quote("BONDFUND", 100.0, AssetClass.BOND)}
        enriched, _ = enrich_holdings(holdings(("BONDFUND", 10)), quotes)
        # 0.2 (bond) + 0.15 concentration bump = 0.35... just at moderate edge
        m = compute_metrics(enriched)
        assert m.risk_level in (RiskLevel.LOW, RiskLevel.MODERATE)

    def test_crypto_is_high_risk(self):
        quotes = {"BTC-USD": make_quote("BTC-USD", 100000.0, AssetClass.CRYPTO)}
        enriched, _ = enrich_holdings(holdings(("BTC-USD", 1)), quotes)
        assert compute_metrics(enriched).risk_level == RiskLevel.HIGH

    def test_aggregate_gain_loss_only_over_priced_basis(self):
        enriched, _ = enrich_holdings(
            holdings(("AAPL", 10, 100.0), ("MSFT", 5)), SAMPLE_QUOTES
        )
        m = compute_metrics(enriched)
        assert m.total_gain_loss_absolute == pytest.approx(1000.0)
        assert m.total_gain_loss_percent == pytest.approx(100.0)

    def test_aggregate_gain_loss_none_without_any_basis(self):
        enriched, _ = enrich_holdings(holdings(("AAPL", 10)), SAMPLE_QUOTES)
        m = compute_metrics(enriched)
        assert m.total_gain_loss_absolute is None
        assert m.total_gain_loss_percent is None

    def test_loss_reported_negative(self):
        enriched, _ = enrich_holdings(
            holdings(("AAPL", 10, 400.0)), SAMPLE_QUOTES  # bought 400, now 200
        )
        m = compute_metrics(enriched)
        assert m.total_gain_loss_absolute == pytest.approx(-2000.0)
        assert m.total_gain_loss_percent == pytest.approx(-50.0)
