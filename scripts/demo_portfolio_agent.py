"""Manual demo of the Portfolio Analysis Agent with live market data and a
real LLM. Requires OPENAI_API_KEY in .env.

Run from the project root:
    .venv/bin/python scripts/demo_portfolio_agent.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.portfolio_analysis.agent import PortfolioAnalysisAgent
from src.agents.portfolio_analysis.model import HoldingInput, PortfolioInput
from src.core.llm import get_llm
from src.data.market_data import MarketDataService


def main() -> None:
    agent = PortfolioAnalysisAgent(llm=get_llm(), market_data=MarketDataService())

    portfolio = PortfolioInput(holdings=[
        HoldingInput(ticker="AAPL", quantity=10, cost_basis_per_share=150),
        HoldingInput(ticker="VOO", quantity=5),
        HoldingInput(ticker="BND", quantity=20),
    ])

    result = agent.run(portfolio)

    m = result.metrics
    print(f"\n{'=' * 60}\nPORTFOLIO ANALYSIS  (data as of {result.as_of:%Y-%m-%d %H:%M UTC})\n{'=' * 60}")
    print(f"Total value:      ${m.total_value:,.2f}")
    print(f"Diversification:  {m.diversification_score}/100")
    print(f"Risk level:       {m.risk_level.value}")
    for s in m.allocation_by_ticker:
        print(f"  {s.label:<6} ${s.value:>12,.2f}  {s.percent:5.1f}%")
    if m.total_gain_loss_absolute is not None:
        print(f"Gain/loss:        ${m.total_gain_loss_absolute:,.2f} "
              f"({m.total_gain_loss_percent:+.1f}%)")
    if result.failed_tickers:
        print(f"Unavailable:      {', '.join(result.failed_tickers)}")
    print(f"\n--- AI Explanation ---\n{result.narrative}")
    print(f"\n{result.disclaimer}")


if __name__ == "__main__":
    main()
