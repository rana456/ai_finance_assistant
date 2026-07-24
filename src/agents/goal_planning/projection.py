"""
Projection engine: deterministic future-value math.

Standard future-value with regular contributions, compounded monthly:

    FV = PV*(1+r)^n + PMT * [((1+r)^n - 1) / r]     r = monthly rate, n = months

Plus the reverse — the monthly contribution required to reach a target — and an
inflation adjustment to express the base case in today's dollars. No LLM, no
randomness: same inputs always give the same numbers.
"""

from __future__ import annotations

from src.agents.goal_planning.model import (
    AllocationFramework,
    GoalAssessment,
    Projection,
    YearPoint,
)

# Assumed long-run inflation for the today's-dollars conversion.
DEFAULT_INFLATION = 2.5  # percent per year


def future_value(pv: float, pmt_monthly: float, annual_return_pct: float, years: int) -> float:
    """FV of a starting balance plus monthly contributions, compounded monthly."""
    n = years * 12
    r = (annual_return_pct / 100) / 12
    if r == 0:
        return pv + pmt_monthly * n
    growth = (1 + r) ** n
    return pv * growth + pmt_monthly * ((growth - 1) / r)


def required_monthly_contribution(
    pv: float, target: float, annual_return_pct: float, years: int
) -> float:
    """Monthly contribution needed to reach `target`, given a starting balance.

    Returns 0 if the starting balance alone already reaches the target.
    """
    n = years * 12
    r = (annual_return_pct / 100) / 12
    if r == 0:
        needed = (target - pv) / n
        return max(0.0, needed)
    growth = (1 + r) ** n
    fv_of_pv = pv * growth
    if fv_of_pv >= target:
        return 0.0
    return (target - fv_of_pv) * r / (growth - 1)


def _real_value(nominal: float, years: int, inflation_pct: float) -> float:
    """Convert a future nominal amount into today's purchasing power."""
    return nominal / ((1 + inflation_pct / 100) ** years)


def build_projection(
    pv: float,
    pmt_monthly: float,
    allocation: AllocationFramework,
    years: int,
    inflation_pct: float = DEFAULT_INFLATION,
) -> Projection:
    """Project low/base/high nominal values, a today's-dollars base value, and a
    year-by-year base-case series."""
    nominal_low = future_value(pv, pmt_monthly, allocation.assumed_return_low, years)
    nominal_base = future_value(pv, pmt_monthly, allocation.assumed_return_base, years)
    nominal_high = future_value(pv, pmt_monthly, allocation.assumed_return_high, years)

    total_contributions = pv + pmt_monthly * years * 12

    year_by_year = [
        YearPoint(
            year=y,
            contributed=pv + pmt_monthly * 12 * y,
            balance=future_value(pv, pmt_monthly, allocation.assumed_return_base, y),
        )
        for y in range(1, years + 1)
    ]

    return Projection(
        nominal_low=nominal_low,
        nominal_base=nominal_base,
        nominal_high=nominal_high,
        real_base=_real_value(nominal_base, years, inflation_pct),
        total_contributions=total_contributions,
        total_growth_base=nominal_base - total_contributions,
        year_by_year=year_by_year,
    )


def assess_goal(
    pv: float,
    pmt_monthly: float,
    target: float,
    allocation: AllocationFramework,
    years: int,
    projection: Projection,
) -> GoalAssessment:
    """Compare the base projection to the target and compute the contribution
    needed to close any gap."""
    projected = projection.nominal_base
    return GoalAssessment(
        target_amount=target,
        projected_base=projected,
        on_track=projected >= target,
        gap_amount=target - projected,
        required_monthly_contribution=required_monthly_contribution(
            pv, target, allocation.assumed_return_base, years
        ),
    )
