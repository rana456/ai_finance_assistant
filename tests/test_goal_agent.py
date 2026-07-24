"""Tests for the Goal Planning Agent: guards, echo-back, plan path, concept path."""

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src.agents.base import EDUCATIONAL_DISCLAIMER
from src.agents.goal_planning.agent import GoalPlanningAgent
from src.agents.goal_planning.model import (
    DrawdownReaction,
    GoalFlexibility,
    GoalPlanRequest,
    RiskAnswers,
    RiskTolerance,
)
from src.rag.chunker import chunk_documents
from src.rag.loader import load_documents
from src.rag.retriever import HybridRetriever
from src.rag.vector_store import VectorStore


class ExplodingLLM(FakeListChatModel):
    def _generate(self, *a, **k):
        raise RuntimeError("outage")


@pytest.fixture
def retriever(fake_embedder):
    chunks = chunk_documents(load_documents())
    store = VectorStore.build(chunks, fake_embedder.embed_documents([c.text for c in chunks]))
    return HybridRetriever(store, fake_embedder, grounding_threshold=0.05)


def make_agent(retriever=None, responses=None, llm_cls=FakeListChatModel):
    llm = llm_cls(responses=responses or ["A clear goal explanation."])
    return GoalPlanningAgent(llm=llm, retriever=retriever)


class TestGuard:
    @pytest.mark.parametrize("q", [
        "Can you guarantee I'll have enough for retirement?",
        "Tell me exactly what stocks to buy for my goal",
        "Which fund should I buy to reach my goal?",
    ])
    def test_guarded_questions_refused(self, q):
        result = make_agent().run(GoalPlanRequest(time_horizon_years=20, question=q))
        assert result.refused
        assert result.refusal_reason
        assert "can't guarantee" in result.narrative.lower()


class TestEchoBack:
    def test_missing_horizon_asks(self):
        result = make_agent().run(GoalPlanRequest(monthly_contribution=200))
        assert result.needs_confirmation
        assert "years" in result.confirmation_prompt.lower()

    def test_missing_risk_presents_questionnaire(self):
        result = make_agent().run(GoalPlanRequest(
            time_horizon_years=20, monthly_contribution=200))
        assert result.needs_confirmation
        assert result.questions is not None
        assert len(result.questions) == 3

    def test_no_money_to_project_asks(self):
        result = make_agent().run(GoalPlanRequest(
            time_horizon_years=20, risk_tolerance=RiskTolerance.MODERATE))
        assert result.needs_confirmation
        assert "starting amount" in result.confirmation_prompt.lower()


class TestPlanPath:
    def _req(self, **kw):
        base = dict(time_horizon_years=30, current_savings=10000,
                    monthly_contribution=500, risk_tolerance=RiskTolerance.MODERATE)
        base.update(kw)
        return GoalPlanRequest(**base)

    def test_projection_produced(self):
        result = make_agent().run(self._req())
        assert result.projection is not None
        assert result.allocation is not None
        assert result.projection.nominal_base > result.projection.total_contributions
        assert result.narrative == "A clear goal explanation."

    def test_assumptions_always_listed(self):
        result = make_agent().run(self._req())
        assert len(result.assumptions) >= 3
        assert any("not guaranteed" in a for a in result.assumptions)

    def test_target_produces_assessment(self):
        result = make_agent().run(self._req(target_amount=1_000_000))
        assert result.assessment is not None
        assert result.assessment.target_amount == 1_000_000

    def test_risk_derived_from_questionnaire(self):
        result = make_agent().run(self._req(
            risk_tolerance=None,
            risk_answers=RiskAnswers(drawdown_reaction=DrawdownReaction.BUY_MORE,
                                     goal_flexibility=GoalFlexibility.FLEXIBLE),
        ))
        # long horizon + aggressive answers -> aggressive framework
        assert result.risk_tolerance == RiskTolerance.AGGRESSIVE

    def test_short_horizon_caps_framework(self):
        result = make_agent().run(self._req(time_horizon_years=2,
                                            risk_tolerance=RiskTolerance.AGGRESSIVE))
        assert result.allocation.risk_tolerance == RiskTolerance.CONSERVATIVE

    def test_disclaimer_present(self):
        assert make_agent().run(self._req()).disclaimer == EDUCATIONAL_DISCLAIMER

    def test_narration_failure_degrades_to_template(self):
        result = make_agent(llm_cls=ExplodingLLM).run(self._req(target_amount=5000))
        assert "temporarily unavailable" in result.narrative
        assert result.projection is not None  # numbers still delivered


class TestConceptPath:
    def test_concept_routes_to_rag(self, retriever):
        result = make_agent(retriever=retriever).run(GoalPlanRequest(
            time_horizon_years=20,
            question="What is modern portfolio theory?"))
        assert len(result.citations) >= 1
        assert result.projection is None

    def test_concept_without_retriever_degrades(self):
        result = make_agent(retriever=None).run(GoalPlanRequest(
            time_horizon_years=20, question="Explain diversification"))
        assert result.citations == []
        assert result.narrative
