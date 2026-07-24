"""Tests for risk questionnaire scoring and the allocation mapper."""

import pytest

from src.agents.goal_planning.allocation import map_allocation
from src.agents.goal_planning.model import (
    DrawdownReaction,
    GoalFlexibility,
    RiskAnswers,
    RiskTolerance,
)
from src.agents.goal_planning.risk_assessment import score_risk_tolerance


def answers(reaction, flex):
    return RiskAnswers(drawdown_reaction=reaction, goal_flexibility=flex)


class TestRiskScoring:
    def test_all_conservative_signals(self):
        # short horizon + would sell + fixed goal -> conservative
        r = score_risk_tolerance(2, answers(DrawdownReaction.SELL, GoalFlexibility.FIXED))
        assert r == RiskTolerance.CONSERVATIVE

    def test_all_aggressive_signals(self):
        # long horizon + would buy more + flexible goal -> aggressive
        r = score_risk_tolerance(30, answers(DrawdownReaction.BUY_MORE, GoalFlexibility.FLEXIBLE))
        assert r == RiskTolerance.AGGRESSIVE

    def test_middle_is_moderate(self):
        r = score_risk_tolerance(10, answers(DrawdownReaction.HOLD, GoalFlexibility.SOMEWHAT_FLEXIBLE))
        assert r == RiskTolerance.MODERATE

    def test_longer_horizon_raises_score(self):
        short = score_risk_tolerance(2, answers(DrawdownReaction.HOLD, GoalFlexibility.SOMEWHAT_FLEXIBLE))
        long = score_risk_tolerance(30, answers(DrawdownReaction.HOLD, GoalFlexibility.SOMEWHAT_FLEXIBLE))
        assert long.rank >= short.rank


class TestAllocationMapper:
    def test_frameworks_sum_to_100(self):
        for risk in RiskTolerance:
            fw = map_allocation(risk, 30)
            assert fw.stock_pct + fw.bond_pct + fw.cash_pct == pytest.approx(100)

    def test_aggressive_has_more_stock_than_conservative(self):
        agg = map_allocation(RiskTolerance.AGGRESSIVE, 30)
        con = map_allocation(RiskTolerance.CONSERVATIVE, 30)
        assert agg.stock_pct > con.stock_pct
        assert agg.assumed_return_base > con.assumed_return_base

    def test_returns_ordered_within_framework(self):
        fw = map_allocation(RiskTolerance.MODERATE, 15)
        assert fw.assumed_return_low < fw.assumed_return_base < fw.assumed_return_high

    def test_short_horizon_caps_aggressive_to_conservative(self):
        fw = map_allocation(RiskTolerance.AGGRESSIVE, 2)
        assert fw.risk_tolerance == RiskTolerance.CONSERVATIVE
        assert "recover" in fw.rationale.lower()

    def test_medium_horizon_caps_aggressive_to_moderate(self):
        fw = map_allocation(RiskTolerance.AGGRESSIVE, 5)
        assert fw.risk_tolerance == RiskTolerance.MODERATE

    def test_long_horizon_no_cap(self):
        fw = map_allocation(RiskTolerance.AGGRESSIVE, 25)
        assert fw.risk_tolerance == RiskTolerance.AGGRESSIVE

    def test_cap_never_raises_risk(self):
        # A conservative tolerance stays conservative even on a long horizon.
        fw = map_allocation(RiskTolerance.CONSERVATIVE, 40)
        assert fw.risk_tolerance == RiskTolerance.CONSERVATIVE
