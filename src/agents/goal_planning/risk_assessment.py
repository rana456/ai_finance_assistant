"""
Risk assessment: a short, deterministic questionnaire -> RiskTolerance.

Three standard axes: time horizon (how long the money can stay invested),
drawdown reaction (behavioral tolerance for volatility), and goal flexibility
(how much room there is to adjust). Each contributes points; the total maps to
a tolerance bucket. Deterministic so it's fully testable and explainable.
"""

from __future__ import annotations

from src.agents.goal_planning.model import (
    DrawdownReaction,
    GoalFlexibility,
    RiskAnswers,
    RiskTolerance,
)

# The questions surfaced to the user when tolerance is unknown. Horizon is
# collected as a numeric field on the request; these are the other two.
QUESTIONNAIRE = [
    "1. Time horizon: how many years until you need this money?",
    "2. If your investments dropped 20% in a few months, would you: "
    "(a) sell to prevent further losses, (b) hold and wait it out, or "
    "(c) buy more while prices are low?",
    "3. How flexible is this goal? Is the target amount and date "
    "(a) fixed, (b) somewhat flexible, or (c) flexible?",
]

_HORIZON_POINTS = [(3, 0), (7, 1), (20, 2)]  # < 3y:0, <7y:1, <20y:2, else 3

_DRAWDOWN_POINTS = {
    DrawdownReaction.SELL: 0,
    DrawdownReaction.HOLD: 1,
    DrawdownReaction.BUY_MORE: 2,
}

_FLEX_POINTS = {
    GoalFlexibility.FIXED: 0,
    GoalFlexibility.SOMEWHAT_FLEXIBLE: 1,
    GoalFlexibility.FLEXIBLE: 2,
}


def _horizon_points(years: int) -> int:
    for threshold, pts in _HORIZON_POINTS:
        if years < threshold:
            return pts
    return 3


def score_risk_tolerance(horizon_years: int, answers: RiskAnswers) -> RiskTolerance:
    """Map (horizon + questionnaire answers) to a risk tolerance bucket.

    Score ranges 0..7:
      0-2 -> conservative, 3-4 -> moderate, 5-7 -> aggressive
    """
    score = (
        _horizon_points(horizon_years)
        + _DRAWDOWN_POINTS[answers.drawdown_reaction]
        + _FLEX_POINTS[answers.goal_flexibility]
    )
    if score <= 2:
        return RiskTolerance.CONSERVATIVE
    if score <= 4:
        return RiskTolerance.MODERATE
    return RiskTolerance.AGGRESSIVE
