"""Tests for the Finance Q&A Agent: advice guard, grounding guard, citations,
LLM fallback. Fake LLM + fake embedder — no network, no API key."""

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src.agents.base import EDUCATIONAL_DISCLAIMER
from src.agents.finance_qa.agent import FinanceQAAgent
from src.agents.finance_qa.model import FinanceQuestion
from src.rag.chunker import chunk_documents
from src.rag.loader import load_documents
from src.rag.retriever import HybridRetriever, RetrievalResult
from src.rag.vector_store import VectorStore


class ExplodingLLM(FakeListChatModel):
    def _generate(self, *args, **kwargs):
        raise RuntimeError("simulated API outage")


class StubRetriever:
    """Returns a fixed retrieval outcome, to test the agent's guards in
    isolation from embedder/BM25 behavior."""

    def __init__(self, result: RetrievalResult):
        self._result = result

    def retrieve(self, query, top_k=4, category=None) -> RetrievalResult:
        return self._result


@pytest.fixture
def retriever(fake_embedder):
    chunks = chunk_documents(load_documents())
    store = VectorStore.build(chunks, fake_embedder.embed_documents([c.text for c in chunks]))
    return HybridRetriever(store, fake_embedder, grounding_threshold=0.05)


def make_agent(retriever, responses=None):
    llm = FakeListChatModel(responses=responses or ["A clear, grounded explanation."])
    return FinanceQAAgent(llm=llm, retriever=retriever)


class TestAdviceGuard:
    @pytest.mark.parametrize("q", [
        "Should I buy Tesla stock?",
        "should i sell my index funds",
        "Which stocks should I invest in?",
        "Can you recommend a good ETF?",
    ])
    def test_advice_questions_refused(self, retriever, q):
        result = make_agent(retriever).run(FinanceQuestion(question=q))
        assert result.refused
        assert result.refusal_reason
        assert result.citations == []
        assert "can't give personalized" in result.answer.lower()

    @pytest.mark.parametrize("q", [
        "What is a stock?",
        "How does compound interest work?",
        "What does diversification mean?",
    ])
    def test_educational_questions_not_refused(self, retriever, q):
        result = make_agent(retriever).run(FinanceQuestion(question=q))
        assert not result.refused


class TestGroundingGuard:
    def test_ungrounded_query_declined_gracefully(self):
        # Stub retrieval as "nothing relevant" to isolate the grounding guard.
        stub = StubRetriever(RetrievalResult(results=[], is_grounded=False))
        agent = make_agent(stub)
        result = agent.run(FinanceQuestion(question="what is the airspeed of a swallow"))
        assert not result.is_grounded
        assert not result.refused
        assert result.citations == []
        assert "knowledge base" in result.answer.lower()

    def test_grounded_query_answered_with_citations(self, retriever):
        agent = make_agent(retriever)
        result = agent.run(FinanceQuestion(question="what is compound interest"))
        assert result.is_grounded
        assert result.answer == "A clear, grounded explanation."
        assert len(result.citations) >= 1
        assert all(c.source for c in result.citations)
        assert all(0 <= c.relevance_score <= 1 for c in result.citations)


class TestGuaranteedContracts:
    def test_disclaimer_always_present(self, retriever):
        for q in ["what is a stock", "should i buy AAPL", "tell me about zxqw"]:
            result = make_agent(retriever).run(FinanceQuestion(question=q))
            assert result.disclaimer == EDUCATIONAL_DISCLAIMER

    def test_citation_snippet_truncated(self, retriever):
        result = make_agent(retriever).run(FinanceQuestion(question="what is a stock"))
        assert all(len(c.snippet) <= 300 for c in result.citations)


class TestLLMFallback:
    def test_llm_failure_degrades_to_template(self, retriever):
        agent = FinanceQAAgent(llm=ExplodingLLM(responses=["unused"]), retriever=retriever)
        result = agent.run(FinanceQuestion(question="what is compound interest"))
        assert result.is_grounded  # still grounded; only the prose degraded
        assert "temporarily unavailable" in result.answer
        assert len(result.citations) >= 1  # citations still attached
