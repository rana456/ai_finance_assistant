"""Tests for the Market Analysis Agent: guards, routing, and narration.
Offline via fake fetchers + fake LLM."""

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src.agents.base import EDUCATIONAL_DISCLAIMER
from src.agents.market_analysis.agent import MarketAnalysisAgent
from src.agents.market_analysis.model import AnalysisType, MarketQuery, Period, TrendSignal
from src.data.news_client import AlphaVantageNewsClient
from src.rag.chunker import chunk_documents
from src.rag.loader import load_documents
from src.rag.retriever import HybridRetriever
from src.rag.vector_store import VectorStore


class ExplodingLLM(FakeListChatModel):
    def _generate(self, *a, **k):
        raise RuntimeError("simulated outage")


@pytest.fixture
def retriever(fake_embedder):
    chunks = chunk_documents(load_documents())
    store = VectorStore.build(chunks, fake_embedder.embed_documents([c.text for c in chunks]))
    return HybridRetriever(store, fake_embedder, grounding_threshold=0.05)


def make_agent(service, retriever=None, news_client=None, responses=None, llm_cls=FakeListChatModel):
    llm = llm_cls(responses=responses or ["A clear market explanation."])
    return MarketAnalysisAgent(llm, service, retriever=retriever, news_client=news_client)


class TestPredictionGuard:
    @pytest.mark.parametrize("q", [
        "Will AAPL go up next week?",
        "Should I buy Tesla?",
        "Is NVDA a good buy right now?",
        "What's the price target for AMZN next month?",
    ])
    def test_prediction_questions_refused(self, market_analysis_service, q):
        result = make_agent(market_analysis_service).run(MarketQuery(question=q))
        assert result.refused
        assert result.refusal_reason
        assert "can't predict" in result.narrative.lower()


class TestDataRouting:
    def test_snapshot(self, market_analysis_service):
        agent = make_agent(market_analysis_service)
        result = agent.run(MarketQuery(question="AAPL price",
                                       tickers=["AAPL"], analysis_type=AnalysisType.SNAPSHOT))
        assert result.intent == AnalysisType.SNAPSHOT
        assert result.snapshots[0].ticker == "AAPL"
        assert result.snapshots[0].price == pytest.approx(200.0)
        assert result.as_of is not None

    def test_metric_populates_valuations(self, market_analysis_service):
        agent = make_agent(market_analysis_service)
        result = agent.run(MarketQuery(question="AAPL P/E",
                                       tickers=["AAPL"], analysis_type=AnalysisType.METRIC))
        assert result.valuations[0].trailing_pe == 30.0

    def test_trend_uses_history(self, market_analysis_service):
        agent = make_agent(market_analysis_service)
        result = agent.run(MarketQuery(question="AAPL this year",
                                       tickers=["AAPL"], analysis_type=AnalysisType.TREND,
                                       period=Period.ONE_YEAR))
        assert result.trends[0].ticker == "AAPL"
        assert result.trends[0].trend_signal == TrendSignal.UPTREND

    def test_compare_multiple_tickers(self, market_analysis_service):
        agent = make_agent(market_analysis_service)
        result = agent.run(MarketQuery(question="AAPL vs MSFT",
                                       tickers=["AAPL", "MSFT"], analysis_type=AnalysisType.COMPARE,
                                       period=Period.ONE_YEAR))
        assert {s.ticker for s in result.snapshots} == {"AAPL", "MSFT"}
        assert len(result.trends) == 2

    def test_overview_uses_indices(self, market_analysis_service):
        agent = make_agent(market_analysis_service)
        result = agent.run(MarketQuery(question="how's the market",
                                       analysis_type=AnalysisType.OVERVIEW))
        tickers = {s.ticker for s in result.snapshots}
        assert {"^GSPC", "^IXIC", "^DJI"} <= tickers

    def test_failed_ticker_reported(self, market_analysis_service):
        agent = make_agent(market_analysis_service)
        result = agent.run(MarketQuery(question="NOPE price",
                                       tickers=["NOPE"], analysis_type=AnalysisType.SNAPSHOT))
        assert result.failed_tickers == ["NOPE"]
        assert result.snapshots == []

    def test_no_ticker_asks_for_confirmation(self, market_analysis_service):
        agent = make_agent(market_analysis_service)
        result = agent.run(MarketQuery(question="how is it doing",
                                       tickers=[], analysis_type=AnalysisType.SNAPSHOT))
        assert result.needs_confirmation
        assert result.confirmation_prompt


class TestConceptRouting:
    def test_concept_routes_to_rag_with_citations(self, market_analysis_service, retriever):
        agent = make_agent(market_analysis_service, retriever=retriever)
        result = agent.run(MarketQuery(question="what moves stock prices",
                                       analysis_type=AnalysisType.CONCEPT))
        assert result.intent == AnalysisType.CONCEPT
        assert len(result.citations) >= 1

    def test_concept_without_retriever_degrades(self, market_analysis_service):
        agent = make_agent(market_analysis_service, retriever=None)
        result = agent.run(MarketQuery(question="what moves stock prices",
                                       analysis_type=AnalysisType.CONCEPT))
        assert result.citations == []
        assert result.narrative


class TestNewsRouting:
    def test_news_unavailable_without_client(self, market_analysis_service):
        agent = make_agent(market_analysis_service, news_client=None)
        result = agent.run(MarketQuery(question="AAPL news",
                                       tickers=["AAPL"], analysis_type=AnalysisType.NEWS))
        assert result.news is None
        assert "isn't available" in result.narrative.lower()

    def test_news_populated_with_client(self, market_analysis_service):
        payload = {"feed": [{
            "title": "Apple update", "url": "http://x", "source": "Reuters",
            "summary": "s", "time_published": "20260721T120000",
            "ticker_sentiment": [{"ticker": "AAPL", "ticker_sentiment_score": "0.4",
                                  "ticker_sentiment_label": "Bullish"}],
        }]}
        client = AlphaVantageNewsClient(api_key="KEY", http_get=lambda url: payload)
        agent = make_agent(market_analysis_service, news_client=client)
        result = agent.run(MarketQuery(question="AAPL news",
                                       tickers=["AAPL"], analysis_type=AnalysisType.NEWS))
        assert result.news is not None
        assert len(result.news.articles) == 1


class TestContracts:
    def test_disclaimer_always_present(self, market_analysis_service, retriever):
        agent = make_agent(market_analysis_service, retriever=retriever)
        for q in [
            MarketQuery(question="Will AAPL crash?"),
            MarketQuery(question="AAPL price", tickers=["AAPL"], analysis_type=AnalysisType.SNAPSHOT),
            MarketQuery(question="what moves prices", analysis_type=AnalysisType.CONCEPT),
        ]:
            assert agent.run(q).disclaimer == EDUCATIONAL_DISCLAIMER

    def test_narration_failure_degrades_to_template(self, market_analysis_service):
        agent = make_agent(market_analysis_service, llm_cls=ExplodingLLM)
        result = agent.run(MarketQuery(question="AAPL price",
                                       tickers=["AAPL"], analysis_type=AnalysisType.SNAPSHOT))
        assert "temporarily unavailable" in result.narrative
        assert "AAPL" in result.narrative
        assert result.snapshots[0].price == pytest.approx(200.0)
