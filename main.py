from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src.agents.portfolio_analysis.agent import PortfolioAnalysisAgent
from src.agents.portfolio_analysis.model import AssetClass, HoldingInput, PortfolioInput


@dataclass(frozen=True)
class _Quote:
    ticker: str
    price: float
    currency: str = "USD"
    asset_class: AssetClass = AssetClass.EQUITY
    sector: str | None = None
    as_of: datetime = datetime.now(timezone.utc)


@dataclass
class _QuoteResult:
    quotes: dict[str, _Quote]
    failed: list[str]


class _StubMarketDataService:
    def __init__(self, quotes: dict[str, _Quote]):
        self._quotes = quotes

    def get_quotes(self, tickers: list[str]) -> _QuoteResult:
        quotes: dict[str, _Quote] = {}
        failed: list[str] = []

        for t in tickers:
            t = t.upper()
            q = self._quotes.get(t)
            if q is None:
                failed.append(t)
            else:
                quotes[t] = q

        return _QuoteResult(quotes=quotes, failed=failed)


def main() -> None:
    llm = FakeListChatModel(responses=["Mock narrative: here is what your portfolio means."])

    quotes = {
        "AAPL": _Quote(
            ticker="AAPL",
            price=200.0,
            asset_class=AssetClass.EQUITY,
            sector="Technology",
        ),
        "VOO": _Quote(
            ticker="VOO",
            price=500.0,
            asset_class=AssetClass.ETF,
            sector=None,
        ),
        "BND": _Quote(
            ticker="BND",
            price=75.0,
            asset_class=AssetClass.BOND,
            sector=None,
        ),
    }

    market_data = _StubMarketDataService(quotes)
    agent = PortfolioAnalysisAgent(llm=llm, market_data=market_data)  # type: ignore[arg-type]

    portfolio = PortfolioInput(
        holdings=[
            HoldingInput(ticker="AAPL", quantity=10, cost_basis_per_share=150.0),
            HoldingInput(ticker="VOO", quantity=2),
            HoldingInput(ticker="BND", quantity=20),
            HoldingInput(ticker="FAKETICK", quantity=1),
        ]
    )

    result = agent.run(portfolio)

    print("\n=== Portfolio Agent Smoke Test ===")
    print(f"As of: {result.as_of.isoformat()}")
    print(f"Failed tickers: {result.failed_tickers}")

    m = result.metrics
    print(f"Total value: ${m.total_value:,.2f}")
    print(f"Diversification score: {m.diversification_score}/100")
    print(f"Risk level: {m.risk_level.value}")

    print("\nAllocation by ticker:")
    for s in m.allocation_by_ticker:
        print(f"- {s.label}: ${s.value:,.2f} ({s.percent:.1f}%)")

    if m.total_gain_loss_absolute is not None:
        print(
            "\nGain/Loss (only holdings with cost basis): "
            f"${m.total_gain_loss_absolute:,.2f} ({m.total_gain_loss_percent:+.1f}%)"
        )

    print("\nNarrative:")
    print(result.narrative)

    print("\nDisclaimer:")
    print(result.disclaimer)


if __name__ == "__main__":
    main()
