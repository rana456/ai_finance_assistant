"""
Pydantic models for the Goal Planning Agent.

Same discipline as the other agents: deterministic engines fill the numeric
models, the LLM only narrates. Because this agent maps a user's situation to an
allocation — which borders on advice — every result carries an explicit
`assumptions` list and a disclaimer, and allocations are framed as illustrative
common frameworks, never personal directives.
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from src.agents.finance_qa.model import Citation, KnowledgeLevel


class GoalType(str, Enum):
    RETIREMENT = "retirement"
    HOUSE = "house"
    EDUCATION = "education"
    EMERGENCY_FUND = "emergency_fund"
    CUSTOM = "custom"


class RiskTolerance(str, Enum):
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"

    @property
    def rank(self) -> int:
        return {"conservative": 0, "moderate": 1, "aggressive": 2}[self.value]


class DrawdownReaction(str, Enum):
    """How the user says they'd react to a sudden 20% portfolio drop."""
    SELL = "sell"          # would sell to stop the losses
    HOLD = "hold"          # would hold and wait it out
    BUY_MORE = "buy_more"  # would buy more at lower prices


class GoalFlexibility(str, Enum):
    """How flexible the goal's timing/amount is."""
    FIXED = "fixed"
    SOMEWHAT_FLEXIBLE = "somewhat_flexible"
    FLEXIBLE = "flexible"


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------


class RiskAnswers(BaseModel):
    """The two questionnaire answers beyond time horizon (which comes from the
    request). Together with horizon they determine risk tolerance."""
    drawdown_reaction: DrawdownReaction
    goal_flexibility: GoalFlexibility


class GoalPlanRequest(BaseModel):
    """A goal-planning request. Supports both a projection (the numeric fields)
    and a conceptual question (the `question` field)."""

    goal_type: GoalType = GoalType.CUSTOM
    target_amount: Optional[float] = Field(None, gt=0, description="The goal amount ($).")
    current_savings: float = Field(0, ge=0, description="Starting balance (PV).")
    monthly_contribution: Optional[float] = Field(
        None, ge=0, description="Regular monthly contribution (PMT)."
    )
    time_horizon_years: Optional[int] = Field(
        None, gt=0, le=100, description="Years until the goal."
    )
    risk_tolerance: Optional[RiskTolerance] = Field(
        None, description="If omitted, derived from the questionnaire."
    )
    risk_answers: Optional[RiskAnswers] = Field(
        None, description="Questionnaire answers used to derive risk tolerance."
    )
    knowledge_level: KnowledgeLevel = KnowledgeLevel.BEGINNER
    question: Optional[str] = Field(
        None, description="Optional natural-language question (concept path)."
    )


# ---------------------------------------------------------------------------
# Output sections
# ---------------------------------------------------------------------------


class AllocationFramework(BaseModel):
    """An illustrative asset-class mix tied to a risk level and horizon."""

    risk_tolerance: RiskTolerance
    stock_pct: float = Field(..., ge=0, le=100)
    bond_pct: float = Field(..., ge=0, le=100)
    cash_pct: float = Field(..., ge=0, le=100)
    assumed_return_low: float = Field(..., description="Annual %, pessimistic.")
    assumed_return_base: float = Field(..., description="Annual %, expected.")
    assumed_return_high: float = Field(..., description="Annual %, optimistic.")
    rationale: str


class YearPoint(BaseModel):
    """One year of the base-case projection, for charting."""
    year: int
    contributed: float = Field(..., description="Cumulative contributions to date.")
    balance: float = Field(..., description="Projected nominal balance (base case).")


class Projection(BaseModel):
    """Future-value projection across scenarios, nominal and inflation-adjusted."""

    nominal_low: float
    nominal_base: float
    nominal_high: float
    real_base: float = Field(..., description="Base case in today's dollars.")
    total_contributions: float
    total_growth_base: float = Field(..., description="nominal_base - total_contributions.")
    year_by_year: list[YearPoint] = Field(default_factory=list)


class GoalAssessment(BaseModel):
    """Whether the plan reaches the target, and what it would take."""

    target_amount: float
    projected_base: float
    on_track: bool
    gap_amount: float = Field(..., description="target - projected_base; positive = shortfall.")
    required_monthly_contribution: float = Field(
        ..., description="Monthly contribution to reach the target at the base return."
    )


# ---------------------------------------------------------------------------
# Top-level result
# ---------------------------------------------------------------------------


class GoalPlanResult(BaseModel):
    """Everything the agent returns. Sections are populated per the path taken."""

    allocation: Optional[AllocationFramework] = None
    projection: Optional[Projection] = None
    assessment: Optional[GoalAssessment] = None
    risk_tolerance: Optional[RiskTolerance] = None
    narrative: str
    citations: list[Citation] = Field(default_factory=list)
    assumptions: list[str] = Field(
        default_factory=list,
        description="Every assumption made (returns, inflation, compounding).",
    )
    disclaimer: str
    needs_confirmation: bool = False
    confirmation_prompt: Optional[str] = None
    questions: Optional[list[str]] = Field(
        None, description="Risk questionnaire to present when tolerance is unknown."
    )
    refused: bool = False
    refusal_reason: Optional[str] = None

    @model_validator(mode="after")
    def refusal_needs_reason(self) -> "GoalPlanResult":
        if self.refused and not self.refusal_reason:
            raise ValueError("A refused result must include a refusal_reason.")
        return self
