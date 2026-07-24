"""Tests for portfolio input models: validation, normalization, merging."""

from datetime import date, timedelta

import pytest
from pydantic import ValidationError

from src.agents.portfolio_analysis.model import (
    HoldingInput,
    InputSource,
    PortfolioInput,
)


class TestHoldingInput:
    def test_ticker_normalized_to_uppercase(self):
        h = HoldingInput(ticker="  aapl ", quantity=10)
        assert h.ticker == "AAPL"

    @pytest.mark.parametrize("ticker", ["BRK.B", "BF-B", "VOO", "GOOG"])
    def test_valid_ticker_formats_accepted(self, ticker):
        assert HoldingInput(ticker=ticker, quantity=1).ticker == ticker

    @pytest.mark.parametrize("ticker", ["AAPL!", "AA PL", "$MSFT", "   "])
    def test_invalid_ticker_formats_rejected(self, ticker):
        with pytest.raises(ValidationError):
            HoldingInput(ticker=ticker, quantity=1)

    @pytest.mark.parametrize("qty", [0, -5])
    def test_non_positive_quantity_rejected(self, qty):
        with pytest.raises(ValidationError):
            HoldingInput(ticker="AAPL", quantity=qty)

    def test_fractional_shares_allowed(self):
        assert HoldingInput(ticker="AAPL", quantity=0.5).quantity == 0.5

    def test_absurd_quantity_rejected(self):
        with pytest.raises(ValidationError):
            HoldingInput(ticker="AAPL", quantity=2e9)

    def test_future_purchase_date_rejected(self):
        with pytest.raises(ValidationError):
            HoldingInput(
                ticker="AAPL", quantity=1,
                purchase_date=date.today() + timedelta(days=1),
            )

    def test_negative_cost_basis_rejected(self):
        with pytest.raises(ValidationError):
            HoldingInput(ticker="AAPL", quantity=1, cost_basis_per_share=-10)


class TestPortfolioInput:
    def test_empty_portfolio_rejected(self):
        with pytest.raises(ValidationError):
            PortfolioInput(holdings=[])

    def test_duplicate_tickers_merged(self):
        p = PortfolioInput(holdings=[
            HoldingInput(ticker="AAPL", quantity=10),
            HoldingInput(ticker="aapl", quantity=5),
        ])
        assert len(p.holdings) == 1
        assert p.holdings[0].quantity == 15

    def test_merge_blends_cost_basis_weighted(self):
        p = PortfolioInput(holdings=[
            HoldingInput(ticker="AAPL", quantity=10, cost_basis_per_share=100),
            HoldingInput(ticker="AAPL", quantity=10, cost_basis_per_share=200),
        ])
        assert p.holdings[0].cost_basis_per_share == pytest.approx(150)

    def test_merge_drops_partial_cost_basis(self):
        p = PortfolioInput(holdings=[
            HoldingInput(ticker="AAPL", quantity=10, cost_basis_per_share=100),
            HoldingInput(ticker="AAPL", quantity=10),
        ])
        assert p.holdings[0].cost_basis_per_share is None

    def test_nl_source_requires_confirmation(self):
        with pytest.raises(ValidationError, match="confirmed"):
            PortfolioInput(
                holdings=[HoldingInput(ticker="AAPL", quantity=1)],
                source=InputSource.NATURAL_LANGUAGE,
            )

    def test_nl_source_with_confirmation_ok(self):
        p = PortfolioInput(
            holdings=[HoldingInput(ticker="AAPL", quantity=1)],
            source=InputSource.NATURAL_LANGUAGE,
            confirmed_by_user=True,
        )
        assert p.source == InputSource.NATURAL_LANGUAGE
