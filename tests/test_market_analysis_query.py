"""Tests for query understanding: LLM JSON parsing and heuristic fallback."""

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src.agents.market_analysis.model import AnalysisType, Period
from src.agents.market_analysis.query_understanding import QueryUnderstanding


def qu(response: str) -> QueryUnderstanding:
    return QueryUnderstanding(FakeListChatModel(responses=[response]))


class TestLLMParsing:
    def test_parses_clean_json(self):
        got = qu('{"intent":"compare","tickers":["AAPL","MSFT"],"period":"1y","ambiguous":false}').extract("q")
        assert got.intent == AnalysisType.COMPARE
        assert got.tickers == ["AAPL", "MSFT"]
        assert got.period == Period.ONE_YEAR

    def test_strips_code_fences(self):
        got = qu('```json\n{"intent":"snapshot","tickers":["TSLA"],"period":"1d"}\n```').extract("q")
        assert got.intent == AnalysisType.SNAPSHOT
        assert got.tickers == ["TSLA"]

    def test_bad_intent_defaults_snapshot(self):
        got = qu('{"intent":"nonsense","tickers":["AAPL"]}').extract("q")
        assert got.intent == AnalysisType.SNAPSHOT

    def test_unparseable_json_falls_back_to_heuristic(self):
        # Not JSON at all -> heuristic reads the *original question*.
        got = qu("I am not json").extract("Compare AAPL vs MSFT")
        assert got.intent == AnalysisType.COMPARE
        assert set(got.tickers) == {"AAPL", "MSFT"}


class TestHeuristicFallback:
    """Force the heuristic by making the LLM raise."""

    def setup_method(self):
        class Boom(FakeListChatModel):
            def _generate(self, *a, **k):
                raise RuntimeError("down")
        self.agent = QueryUnderstanding(Boom(responses=["x"]))

    def test_detects_trend(self):
        got = self.agent.extract("How has AAPL performed this year?")
        assert got.intent == AnalysisType.TREND
        assert got.period == Period.ONE_YEAR
        assert "AAPL" in got.tickers

    def test_detects_metric(self):
        got = self.agent.extract("What is the P/E of MSFT?")
        assert got.intent == AnalysisType.METRIC

    def test_detects_overview_without_ticker(self):
        got = self.agent.extract("How is the market doing today?")
        assert got.intent == AnalysisType.OVERVIEW

    def test_detects_concept(self):
        got = self.agent.extract("What moves stock prices?")
        assert got.intent == AnalysisType.CONCEPT

    def test_stopwords_not_treated_as_tickers(self):
        got = self.agent.extract("What is AAPL worth?")
        assert got.tickers == ["AAPL"]  # 'IS' filtered out

    def test_dollar_prefixed_ticker(self):
        got = self.agent.extract("snapshot of $NVDA please")
        assert "NVDA" in got.tickers
