"""Manual demo of the Goal Planning Agent with a real LLM.

Requires OPENAI_API_KEY and (for concept questions) a prebuilt index.

    .venv/bin/python scripts/demo_goal_planning.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

from src.agents.goal_planning.agent import GoalPlanningAgent
from src.agents.goal_planning.model import (
    DrawdownReaction,
    GoalFlexibility,
    GoalPlanRequest,
    GoalType,
    RiskAnswers,
)
from src.core.llm import get_llm
from src.rag.embedder import OpenAIEmbedder
from src.rag.retriever import HybridRetriever
from src.rag.vector_store import VectorStore

INDEX_DIR = Path(__file__).resolve().parent.parent / "src" / "data" / "index"


def show(title, result):
    print(f"\n{'=' * 62}\n{title}\n{'=' * 62}")
    if result.needs_confirmation:
        print("[needs confirmation]")
        if result.questions:
            for q in result.questions:
                print("   " + q)
    print(result.narrative)
    if result.allocation:
        a = result.allocation
        print(f"\n  Framework: {a.risk_tolerance.value} — "
              f"{a.stock_pct:.0f}% stocks / {a.bond_pct:.0f}% bonds / {a.cash_pct:.0f}% cash")
    if result.projection:
        p = result.projection
        print(f"  Projected (base): ${p.nominal_base:,.0f}  "
              f"(range ${p.nominal_low:,.0f}–${p.nominal_high:,.0f})")
        print(f"  In today's dollars: ${p.real_base:,.0f}")
    if result.assessment:
        a = result.assessment
        status = "on track" if a.on_track else f"short by ${a.gap_amount:,.0f}"
        print(f"  Goal: {status}; need ~${a.required_monthly_contribution:,.0f}/mo to hit target")
    if result.citations:
        print("  Sources: " + ", ".join(sorted({c.title for c in result.citations})))
    if result.assumptions:
        print("  Assumptions:")
        for a in result.assumptions:
            print(f"    - {a}")


def main() -> None:
    load_dotenv()
    retriever = None
    if INDEX_DIR.exists():
        retriever = HybridRetriever(VectorStore.load(INDEX_DIR), OpenAIEmbedder())
    agent = GoalPlanningAgent(llm=get_llm(), retriever=retriever)

    # 1. Full retirement plan with a target and questionnaire-derived risk.
    show("1. RETIREMENT PLAN (with target)", agent.run(GoalPlanRequest(
        goal_type=GoalType.RETIREMENT,
        target_amount=1_000_000,
        current_savings=25_000,
        monthly_contribution=800,
        time_horizon_years=30,
        risk_answers=RiskAnswers(drawdown_reaction=DrawdownReaction.HOLD,
                                 goal_flexibility=GoalFlexibility.SOMEWHAT_FLEXIBLE),
    )))

    # 2. Concept question -> RAG.
    show("2. CONCEPT: Modern Portfolio Theory", agent.run(GoalPlanRequest(
        time_horizon_years=20, question="What is modern portfolio theory in simple terms?")))

    # 3. Guard.
    show("3. GUARD: guarantee request", agent.run(GoalPlanRequest(
        time_horizon_years=20, question="Can you guarantee I'll be a millionaire?")))


if __name__ == "__main__":
    main()
