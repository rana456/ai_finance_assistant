"""Tests for the Tax Education Agent: advice guard, grounding, citations,
disclaimer. Offline via fake embedder + fake LLM."""

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src.agents.tax_education.agent import TaxEducationAgent
from src.agents.tax_education.model import TAX_DISCLAIMER, TaxQuestion
from src.rag.chunker import chunk_documents
from src.rag.loader import load_documents
from src.rag.retriever import HybridRetriever, RetrievalResult
from src.rag.vector_store import VectorStore


class ExplodingLLM(FakeListChatModel):
    def _generate(self, *a, **k):
        raise RuntimeError("outage")


class StubRetriever:
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
    llm = FakeListChatModel(responses=responses or ["A clear tax explanation."])
    return TaxEducationAgent(llm=llm, retriever=retriever)


class TestAdviceGuard:
    @pytest.mark.parametrize("q", [
        "How much tax will I owe on my bonus?",
        "Should I do Roth or traditional for my situation?",
        "How do I report my capital gains on my return?",
        "What's my marginal tax bracket?",
        "Can I deduct my home office on my taxes?",
    ])
    def test_personalized_tax_questions_refused(self, retriever, q):
        result = make_agent(retriever).run(TaxQuestion(question=q))
        assert result.refused
        assert result.refusal_reason
        assert result.consult_professional
        assert result.citations == []
        assert "tax professional" in result.answer.lower()

    @pytest.mark.parametrize("q", [
        "What is a Roth IRA?",
        "How does an HSA work?",
        "What's the difference between short-term and long-term capital gains?",
        "How are qualified dividends taxed?",
    ])
    def test_educational_questions_not_refused(self, retriever, q):
        result = make_agent(retriever).run(TaxQuestion(question=q))
        assert not result.refused


class TestGrounding:
    def test_ungrounded_declined(self):
        stub = StubRetriever(RetrievalResult(results=[], is_grounded=False))
        result = make_agent(stub).run(TaxQuestion(question="what is the airspeed of a swallow"))
        assert not result.is_grounded
        assert not result.refused
        assert result.citations == []
        assert "knowledge base" in result.answer.lower()

    def test_grounded_answer_has_citations(self, retriever):
        result = make_agent(retriever).run(TaxQuestion(question="how does a roth ira work"))
        assert result.is_grounded
        assert result.answer == "A clear tax explanation."
        assert len(result.citations) >= 1
        assert all(c.source for c in result.citations)


class TestContracts:
    def test_disclaimer_always_present_and_mentions_verification(self, retriever):
        for q in ["what is a 401k", "how much tax will i owe", "explain capital gains"]:
            result = make_agent(retriever).run(TaxQuestion(question=q))
            assert result.disclaimer == TAX_DISCLAIMER
            assert "irs.gov" in result.disclaimer.lower()
            assert "not tax advice" in result.disclaimer.lower()

    def test_llm_failure_degrades_to_template(self, retriever):
        agent = TaxEducationAgent(llm=ExplodingLLM(responses=["x"]), retriever=retriever)
        result = agent.run(TaxQuestion(question="how does an hsa work"))
        assert result.is_grounded
        assert "temporarily unavailable" in result.answer
        assert len(result.citations) >= 1

    def test_tax_corpus_loaded(self):
        docs = {d.doc_id for d in load_documents()}
        assert {"tax-treatments-overview", "tax-hsa-triple-advantage",
                "tax-capital-gains-dividends"} <= docs
