"""
Allocation mapper: (risk tolerance + time horizon) -> illustrative framework.

The base frameworks are mainstream, widely-published asset-class mixes
(consistent across Vanguard/Fidelity/Schwab/Morningstar guidance). Time horizon
*caps* aggressiveness: a short horizon can't ride out a downturn, so even an
aggressive risk tolerance is mapped to a more conservative mix — with the reason
stated in the rationale. These are educational frameworks, not recommendations.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.agents.goal_planning.model import AllocationFramework, RiskTolerance


@dataclass(frozen=True)
class _Framework:
    stock: float
    bond: float
    cash: float
    ret_low: float
    ret_base: float
    ret_high: float


# Base frameworks by tolerance. Return ranges are illustrative historical
# nominal averages (stocks ~7-10%, bonds ~3-5%, cash ~4-5%), blended by mix.
_FRAMEWORKS = {
    RiskTolerance.CONSERVATIVE: _Framework(25, 70, 5, 3.0, 4.0, 5.0),
    RiskTolerance.MODERATE: _Framework(60, 35, 5, 5.0, 6.5, 8.0),
    RiskTolerance.AGGRESSIVE: _Framework(85, 15, 0, 7.0, 8.5, 10.0),
}


def _horizon_cap(horizon_years: int) -> RiskTolerance:
    """The most aggressive framework a given horizon can justify."""
    if horizon_years < 3:
        return RiskTolerance.CONSERVATIVE
    if horizon_years < 7:
        return RiskTolerance.MODERATE
    return RiskTolerance.AGGRESSIVE


def map_allocation(
    risk_tolerance: RiskTolerance, horizon_years: int
) -> AllocationFramework:
    """Build the illustrative framework, applying the horizon cap."""
    cap = _horizon_cap(horizon_years)
    effective = risk_tolerance if risk_tolerance.rank <= cap.rank else cap
    fw = _FRAMEWORKS[effective]

    if effective != risk_tolerance:
        rationale = (
            f"Your answers suggest a {risk_tolerance.value} tolerance, but with "
            f"only {horizon_years} years until the goal there may not be time to "
            f"recover from a downturn — so a more {effective.value} mix is commonly "
            "considered more appropriate for this horizon. This is a general "
            "framework, not a personal recommendation."
        )
    else:
        rationale = (
            f"A {effective.value} framework commonly uses roughly "
            f"{fw.stock:.0f}% stocks / {fw.bond:.0f}% bonds / {fw.cash:.0f}% cash, "
            f"which suits a {horizon_years}-year horizon. This is a general "
            "framework, not a personal recommendation."
        )

    return AllocationFramework(
        risk_tolerance=effective,
        stock_pct=fw.stock,
        bond_pct=fw.bond,
        cash_pct=fw.cash,
        assumed_return_low=fw.ret_low,
        assumed_return_base=fw.ret_base,
        assumed_return_high=fw.ret_high,
        rationale=rationale,
    )
