"""
Goal Planning Agent.

Orchestrates: guarantee/advice guard -> route (concept vs. plan) -> risk
assessment (questionnaire echo-back if needed) -> allocation mapper ->
projection engine -> goal assessment -> LLM narration -> GoalPlanResult.

Guardrails enforced outside the LLM:
- Guarantee/advice guard: "guarantee I'll have enough" / "exactly what should I
  buy" are refused and redirected to education.
- Assumptions are always listed explicitly; projections are illustrations.
- Allocations are framed as common frameworks, never personal directives.
- Concept questions are answered from the cited RAG knowledge base.
"""

from __future__ import annotations

import json
import logging
import re

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.base import BaseFinanceAgent, EDUCATIONAL_DISCLAIMER
from src.agents.finance_qa.model import Citation
from src.agents.goal_planning.allocation import map_allocation
from src.agents.goal_planning.model import (
    GoalPlanRequest,
    GoalPlanResult,
    Projection,
)
from src.agents.goal_planning.projection import (
    DEFAULT_INFLATION,
    assess_goal,
    build_projection,
)
from src.agents.goal_planning.risk_assessment import QUESTIONNAIRE, score_risk_tolerance

logger = logging.getLogger(__name__)

_GUARD_PATTERNS = [
    r"\bguarantee",
    r"\bam i guaranteed\b",
    r"\bwill i (definitely|for sure|certainly) have\b",
    r"\bpromise\b.*\b(enough|return|money)\b",
    r"\b(what|which) (specific )?(stock|fund|etf|share|security)s? (should i|to|do i|can i) (buy|pick|choose|invest in|get)\b",
    r"\btell me (exactly |specifically )?what to (buy|invest in)\b",
]

_GUARD_REFUSAL = (
    "I can't guarantee outcomes or tell you exactly what to buy — future returns "
    "are never certain, and specific investment picks are personalized advice. "
    "What I can do is show you an illustrative projection based on clearly-stated "
    "assumptions, explain how risk tolerance maps to common allocation "
    "frameworks, and walk through the math so you can plan with realistic "
    "expectations. Want me to run a projection for your goal?"
)

_CONCEPT_KEYWORDS = [
    "what is", "what's", "how does", "how do", "explain", "why",
    "modern portfolio", "mpt", "diversification", "risk tolerance",
    "asset allocation", "efficient frontier",
]

_NARRATION_SYSTEM = """You are a financial education assistant explaining a goal \
projection to a beginner. You are given pre-computed numbers as JSON.

Rules:
- Explain ONLY the numbers provided; never compute or invent figures.
- Present the allocation as a common educational framework, NOT a personal \
recommendation, and never name specific securities or funds.
- State that projections are illustrations based on the assumptions given, and \
that actual returns vary and are not guaranteed.
- If there's a gap to the goal, explain the required-contribution figure kindly.
- Define terms simply. Be warm and concise (under 250 words).
- Do not write a disclaimer or list assumptions/sources; those are added \
automatically."""

_CONCEPT_SYSTEM = """You are a financial education assistant. Answer the user's \
question using ONLY the numbered sources provided. Define terms simply, stay \
under 250 words, don't invent facts, and give no personalized advice or \
predictions. No disclaimer or source list — those are added automatically."""


class GoalPlanningAgent(BaseFinanceAgent):
    name = "goal_planning"
    description = (
        "Helps plan financial goals: projects future value of savings with "
        "compound growth, maps risk tolerance and horizon to illustrative "
        "allocation frameworks, and explains the concepts. Educational only — "
        "no guarantees or specific security recommendations."
    )

    def __init__(self, llm: BaseChatModel, retriever=None):
        super().__init__(llm)
        self.retriever = retriever

    def run(self, request: GoalPlanRequest) -> GoalPlanResult:
        # --- Guard ---
        if request.question and self._is_guarded(request.question):
            return GoalPlanResult(
                narrative=_GUARD_REFUSAL, disclaimer=EDUCATIONAL_DISCLAIMER,
                refused=True, refusal_reason="Guarantees and specific picks are out of scope.",
            )

        # --- Concept path ---
        if request.question and self._is_concept(request.question):
            return self._handle_concept(request)

        # --- Plan path ---
        return self._handle_plan(request)

    # --- plan path ---

    def _handle_plan(self, request: GoalPlanRequest) -> GoalPlanResult:
        if request.time_horizon_years is None:
            return self._ask(
                "How many years until you need this money?",
                questions=None,
            )

        # Resolve risk tolerance: explicit > questionnaire > ask.
        risk = request.risk_tolerance
        if risk is None:
            if request.risk_answers is None:
                return self._ask(
                    "To tailor an illustrative framework to you, I need a quick "
                    "read on your risk tolerance.",
                    questions=QUESTIONNAIRE,
                )
            risk = score_risk_tolerance(request.time_horizon_years, request.risk_answers)

        pmt = request.monthly_contribution or 0.0
        if pmt <= 0 and request.current_savings <= 0:
            return self._ask(
                "To project your goal, tell me a starting amount and/or a monthly "
                "contribution to work with.",
                questions=None,
            )

        allocation = map_allocation(risk, request.time_horizon_years)
        projection = build_projection(
            request.current_savings, pmt, allocation, request.time_horizon_years
        )
        assessment = None
        if request.target_amount is not None:
            assessment = assess_goal(
                request.current_savings, pmt, request.target_amount,
                allocation, request.time_horizon_years, projection,
            )

        assumptions = self._assumptions(allocation, request.time_horizon_years)
        narrative = self._narrate_plan(request, allocation, projection, assessment, assumptions)
        return GoalPlanResult(
            allocation=allocation,
            projection=projection,
            assessment=assessment,
            risk_tolerance=allocation.risk_tolerance,
            narrative=narrative,
            assumptions=assumptions,
            disclaimer=EDUCATIONAL_DISCLAIMER,
        )

    def _handle_concept(self, request: GoalPlanRequest) -> GoalPlanResult:
        if self.retriever is None:
            return GoalPlanResult(
                narrative=(
                    "I can run goal projections and explain allocation frameworks, "
                    "but the concept knowledge base isn't available here. Try asking "
                    "me to project a savings goal instead."
                ),
                disclaimer=EDUCATIONAL_DISCLAIMER,
            )
        retrieval = self.retriever.retrieve(request.question, top_k=4)
        if not retrieval.is_grounded:
            return GoalPlanResult(
                narrative=(
                    "I don't have that topic in my knowledge base yet. I can help "
                    "with compound growth, risk tolerance, asset allocation, and "
                    "Modern Portfolio Theory basics — or project a specific goal."
                ),
                disclaimer=EDUCATIONAL_DISCLAIMER,
            )
        narrative = self._narrate_concept(request, retrieval.results)
        return GoalPlanResult(
            narrative=narrative,
            citations=self._citations_from(retrieval.results),
            disclaimer=EDUCATIONAL_DISCLAIMER,
        )

    # --- narration ---

    def _narrate_plan(self, request, allocation, projection, assessment, assumptions) -> str:
        payload = {
            "goal_type": request.goal_type.value,
            "horizon_years": request.time_horizon_years,
            "allocation": allocation.model_dump(),
            "projection": projection.model_dump(exclude={"year_by_year"}),
            "assessment": assessment.model_dump() if assessment else None,
            "assumptions": assumptions,
        }
        user_content = f"Goal question: {request.question or '(projection request)'}\n\nData:\n{json.dumps(payload, default=str)}"
        try:
            resp = self.llm.invoke(
                [SystemMessage(content=_NARRATION_SYSTEM), HumanMessage(content=user_content)]
            )
            return resp.content
        except Exception:
            logger.warning("Plan narration failed; using template", exc_info=True)
            return self._template_plan(allocation, projection, assessment)

    def _narrate_concept(self, request, results) -> str:
        sources = "\n\n".join(
            f"[Source {i}] {sc.chunk.title} ({sc.chunk.source})\n{sc.chunk.text}"
            for i, sc in enumerate(results, start=1)
        )
        user_content = f"Question: {request.question}\n\nSources:\n{sources}"
        try:
            resp = self.llm.invoke(
                [SystemMessage(content=_CONCEPT_SYSTEM), HumanMessage(content=user_content)]
            )
            return resp.content
        except Exception:
            logger.warning("Concept narration failed; using template", exc_info=True)
            return (
                "Here's the most relevant information from my knowledge base "
                f"(AI explanation is temporarily unavailable):\n\n{results[0].chunk.text}"
            )

    @staticmethod
    def _template_plan(allocation, projection, assessment) -> str:
        lines = [
            f"Illustrative {allocation.risk_tolerance.value} framework: "
            f"{allocation.stock_pct:.0f}% stocks / {allocation.bond_pct:.0f}% bonds / "
            f"{allocation.cash_pct:.0f}% cash.",
            f"Projected value (base case): ${projection.nominal_base:,.0f} "
            f"(range ${projection.nominal_low:,.0f}–${projection.nominal_high:,.0f}).",
            f"That's about ${projection.real_base:,.0f} in today's dollars.",
            f"Total contributed: ${projection.total_contributions:,.0f}; "
            f"growth: ${projection.total_growth_base:,.0f}.",
        ]
        if assessment:
            if assessment.on_track:
                lines.append(f"This reaches your ${assessment.target_amount:,.0f} target.")
            else:
                lines.append(
                    f"This falls ${assessment.gap_amount:,.0f} short; reaching the "
                    f"target would take about ${assessment.required_monthly_contribution:,.0f}/month."
                )
        lines.append("(Detailed AI explanation is temporarily unavailable.)")
        return "\n".join(lines)

    # --- helpers ---

    @staticmethod
    def _assumptions(allocation, years: int) -> list[str]:
        return [
            f"Assumed annual returns: {allocation.assumed_return_base:.1f}% base "
            f"({allocation.assumed_return_low:.1f}%–{allocation.assumed_return_high:.1f}% range), "
            "based on illustrative historical averages — not guaranteed.",
            "Returns compounded monthly; contributions assumed constant.",
            f"Today's-dollars figures assume {DEFAULT_INFLATION:.1f}% annual inflation.",
            "This is an educational illustration, not a personalized recommendation.",
        ]

    @staticmethod
    def _is_guarded(question: str) -> bool:
        q = question.lower()
        return any(re.search(p, q) for p in _GUARD_PATTERNS)

    @staticmethod
    def _is_concept(question: str) -> bool:
        q = question.lower()
        return any(k in q for k in _CONCEPT_KEYWORDS)

    @staticmethod
    def _citations_from(results) -> list[Citation]:
        citations = []
        for sc in results:
            snippet = sc.chunk.text
            if len(snippet) > 300:
                snippet = snippet[:297].rstrip() + "..."
            citations.append(Citation(
                source=sc.chunk.source, title=sc.chunk.title, url=sc.chunk.source_url,
                snippet=snippet, relevance_score=round(sc.cosine, 4),
            ))
        return citations

    @staticmethod
    def _ask(prompt: str, questions) -> GoalPlanResult:
        return GoalPlanResult(
            narrative=prompt,
            disclaimer=EDUCATIONAL_DISCLAIMER,
            needs_confirmation=True,
            confirmation_prompt=prompt,
            questions=questions,
        )
