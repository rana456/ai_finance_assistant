"""Tests for the Portfolio Analysis Agent: pipeline wiring, LLM guardrails,
fallback behavior. Uses a fake LLM — no network, no API key."""

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src.agents.base import EDUCATIONAL_DISCLAIMER
from src.agents.portfolio_analysis.agent import PortfolioAnalysisAgent
from src.agents.portfolio_analysis.model import HoldingInput, PortfolioInput


class ExplodingLLM(FakeListChatModel):
    """Fake LLM that always raises, to exercise the fallback path."""

    def _generate(self, *args, **kwargs):
        raise RuntimeError("simulated API outage")


def make_agent(market_data, responses=None) -> PortfolioAnalysisAgent:
    llm = FakeListChatModel(responses=responses or ["A friendly narrative."])
    return PortfolioAnalysisAgent(llm=llm, market_data=market_data)


def portfolio(*specs) -> PortfolioInput:
    return PortfolioInput(holdings=[
        HoldingInput(
            ticker=s[0], quantity=s[1],
            cost_basis_per_share=s[2] if len(s) > 2 else None,
        )
        for s in specs
    ])


class TestPipeline:
    def test_happy_path(self, market_data):
        agent = make_agent(market_data)
        result = agent.run(portfolio(("AAPL", 10), ("VOO", 4)))
        assert result.metrics.total_value == pytest.approx(4000.0)
        assert result.narrative == "A friendly narrative."
        assert result.disclaimer == EDUCATIONAL_DISCLAIMER
        assert result.failed_tickers == []
        assert len(result.holdings) == 2

    def test_partial_failure_still_analyzes(self, market_data):
        agent = make_agent(market_data)
        result = agent.run(portfolio(("AAPL", 10), ("FAKETICK", 5)))
        assert result.failed_tickers == ["FAKETICK"]
        assert result.metrics.total_value == pytest.approx(2000.0)

    def test_all_tickers_failing_raises_friendly_error(self, market_data):
        agent = make_agent(market_data)
        with pytest.raises(ValueError, match="NOPE1"):
            agent.run(portfolio(("NOPE1", 1), ("NOPE2", 2)))

    def test_disclaimer_always_present(self, market_data):
        result = make_agent(market_data).run(portfolio(("AAPL", 1)))
        assert "not financial advice" in result.disclaimer


class TestLLMFallback:
    def test_llm_failure_degrades_to_template(self, market_data):
        agent = PortfolioAnalysisAgent(
            llm=ExplodingLLM(responses=["unused"]), market_data=market_data
        )
        result = agent.run(portfolio(("AAPL", 10, 100.0)))
        # Metrics still delivered, template narrative mentions the numbers
        assert "$2,000.00" in result.narrative
        assert "temporarily unavailable" in result.narrative
        assert result.metrics.total_value == pytest.approx(2000.0)

    def test_template_mentions_failed_tickers(self, market_data):
        agent = PortfolioAnalysisAgent(
            llm=ExplodingLLM(responses=["unused"]), market_data=market_data
        )
        result = agent.run(portfolio(("AAPL", 10), ("FAKETICK", 1)))
        assert "FAKETICK" in result.narrative
