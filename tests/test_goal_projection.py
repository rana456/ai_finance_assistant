"""Tests for the goal projection engine (pure math)."""

import pytest

from src.agents.goal_planning.allocation import map_allocation
from src.agents.goal_planning.model import RiskTolerance
from src.agents.goal_planning.projection import (
    assess_goal,
    build_projection,
    future_value,
    required_monthly_contribution,
)


class TestFutureValue:
    def test_zero_return_is_sum_of_contributions(self):
        # PV 1000 + $100/mo for 10y at 0% = 1000 + 100*120
        assert future_value(1000, 100, 0.0, 10) == pytest.approx(1000 + 12000)

    def test_growth_beats_contributions_at_positive_return(self):
        fv = future_value(0, 100, 7.0, 30)
        contributed = 100 * 12 * 30
        assert fv > contributed  # compounding added growth

    def test_higher_return_gives_higher_fv(self):
        low = future_value(1000, 100, 4.0, 20)
        high = future_value(1000, 100, 8.0, 20)
        assert high > low

    def test_lump_sum_only(self):
        # $10k at 6% for 10y, no contributions, monthly compounding
        fv = future_value(10000, 0, 6.0, 10)
        assert fv == pytest.approx(10000 * (1 + 0.06 / 12) ** 120)


class TestRequiredContribution:
    def test_zero_when_pv_already_exceeds_target(self):
        assert required_monthly_contribution(200000, 100000, 6.0, 10) == 0.0

    def test_plugging_back_reaches_target(self):
        pmt = required_monthly_contribution(5000, 100000, 6.0, 15)
        reached = future_value(5000, pmt, 6.0, 15)
        assert reached == pytest.approx(100000, rel=1e-6)

    def test_zero_return_linear(self):
        # Need 12000 more over 120 months at 0% -> 100/mo
        pmt = required_monthly_contribution(0, 12000, 0.0, 10)
        assert pmt == pytest.approx(100.0)


class TestProjection:
    def _alloc(self):
        return map_allocation(RiskTolerance.MODERATE, 20)

    def test_scenarios_ordered(self):
        p = build_projection(10000, 300, self._alloc(), 20)
        assert p.nominal_low < p.nominal_base < p.nominal_high

    def test_real_less_than_nominal(self):
        p = build_projection(10000, 300, self._alloc(), 20)
        assert p.real_base < p.nominal_base

    def test_growth_is_nominal_minus_contributions(self):
        p = build_projection(10000, 300, self._alloc(), 20)
        assert p.total_growth_base == pytest.approx(p.nominal_base - p.total_contributions)

    def test_year_series_length_and_endpoint(self):
        p = build_projection(10000, 300, self._alloc(), 20)
        assert len(p.year_by_year) == 20
        assert p.year_by_year[-1].balance == pytest.approx(p.nominal_base)
        assert p.year_by_year[-1].year == 20

    def test_total_contributions(self):
        p = build_projection(10000, 300, self._alloc(), 20)
        assert p.total_contributions == pytest.approx(10000 + 300 * 12 * 20)


class TestAssessment:
    def _alloc(self):
        return map_allocation(RiskTolerance.AGGRESSIVE, 30)

    def test_on_track_when_projection_meets_target(self):
        alloc = self._alloc()
        p = build_projection(10000, 500, alloc, 30)
        a = assess_goal(10000, 500, 10000, alloc, 30, p)  # trivially low target
        assert a.on_track
        assert a.gap_amount < 0  # surplus

    def test_shortfall_reports_gap_and_required(self):
        alloc = self._alloc()
        p = build_projection(0, 100, alloc, 30)
        target = p.nominal_base * 2  # deliberately unreachable at current pmt
        a = assess_goal(0, 100, target, alloc, 30, p)
        assert not a.on_track
        assert a.gap_amount > 0
        assert a.required_monthly_contribution > 100
