"""Tests for the market analysis pure calculators."""

import pytest

from src.agents.market_analysis.analyzer import (
    build_analyst_view,
    build_news_sentiment,
    build_snapshot,
    build_trend,
    build_valuation,
)
from src.agents.market_analysis.model import Period, SentimentLabel, TrendSignal

from tests.conftest import make_detailed, _linear
from src.data.market_analysis_service import PriceHistory
from datetime import datetime, timezone


class TestSnapshot:
    def test_change_computed_from_previous_close(self):
        q = make_detailed("AAPL", 200.0, previous_close=190.0)
        s = build_snapshot(q)
        assert s.change_abs == pytest.approx(10.0)
        assert s.change_pct == pytest.approx(10 / 190 * 100)

    def test_volume_vs_average(self):
        q = make_detailed("AAPL", 200.0, volume=60_000_000, avg_volume=40_000_000)
        s = build_snapshot(q)
        assert s.volume_vs_avg_pct == pytest.approx(150.0)

    def test_pct_in_52wk_range(self):
        q = make_detailed("AAPL", 150.0, week52_low=100.0, week52_high=200.0)
        s = build_snapshot(q)
        assert s.pct_in_52wk_range == pytest.approx(50.0)

    def test_pct_in_range_clamped(self):
        q = make_detailed("AAPL", 250.0, week52_low=100.0, week52_high=200.0)
        s = build_snapshot(q)
        assert s.pct_in_52wk_range == 100.0

    def test_missing_previous_close_leaves_change_none(self):
        q = make_detailed("AAPL", 200.0, previous_close=None)
        s = build_snapshot(q)
        assert s.change_abs is None and s.change_pct is None


class TestValuationAndAnalyst:
    def test_valuation_passthrough(self):
        q = make_detailed("AAPL", 200.0, trailing_pe=25.0, beta=1.2)
        v = build_valuation(q)
        assert v.trailing_pe == 25.0 and v.beta == 1.2

    def test_analyst_view_present(self):
        q = make_detailed("AAPL", 200.0)
        assert build_analyst_view(q) is not None

    def test_analyst_view_none_for_etf(self):
        q = make_detailed("VOO", 500.0, recommendation=None, target_mean=None, num_analysts=None)
        assert build_analyst_view(q) is None


class TestTrend:
    def _hist(self, ticker, closes):
        return PriceHistory(ticker=ticker, period="1y", closes=closes,
                            as_of=datetime.now(timezone.utc))

    def test_pct_change_and_range(self):
        t = build_trend(self._hist("X", [100, 110, 120]), Period.ONE_MONTH)
        assert t.pct_change == pytest.approx(20.0)
        assert t.period_low == 100 and t.period_high == 120

    def test_uptrend_via_moving_averages(self):
        t = build_trend(self._hist("AAPL", _linear(150, 200, 250)), Period.ONE_YEAR)
        assert t.sma_50 is not None and t.sma_200 is not None
        assert t.trend_signal == TrendSignal.UPTREND

    def test_downtrend_via_moving_averages(self):
        t = build_trend(self._hist("MSFT", _linear(450, 400, 250)), Period.ONE_YEAR)
        assert t.trend_signal == TrendSignal.DOWNTREND

    def test_short_history_uses_pct_fallback(self):
        # Too few points for SMA-50; large positive move -> uptrend by threshold.
        t = build_trend(self._hist("X", _linear(100, 120, 20)), Period.ONE_MONTH)
        assert t.sma_50 is None
        assert t.trend_signal == TrendSignal.UPTREND

    def test_sideways_small_move(self):
        t = build_trend(self._hist("X", _linear(100, 101, 20)), Period.ONE_MONTH)
        assert t.trend_signal == TrendSignal.SIDEWAYS

    def test_empty_history_raises(self):
        with pytest.raises(ValueError):
            build_trend(self._hist("X", []), Period.ONE_MONTH)


class TestNewsSentiment:
    def _payload(self, score, label):
        return {
            "feed": [{
                "title": "Big news", "url": "http://x", "source": "Reuters",
                "summary": "summary text", "time_published": "20260721T120000",
                "ticker_sentiment": [
                    {"ticker": "AAPL", "ticker_sentiment_score": str(score),
                     "ticker_sentiment_label": label},
                ],
            }]
        }

    def test_parses_article_and_overall(self):
        news = build_news_sentiment("AAPL", self._payload(0.45, "Bullish"))
        assert len(news.articles) == 1
        assert news.articles[0].sentiment_label == SentimentLabel.BULLISH
        assert news.overall_label == SentimentLabel.BULLISH

    def test_bearish_overall(self):
        news = build_news_sentiment("AAPL", self._payload(-0.5, "Bearish"))
        assert news.overall_label == SentimentLabel.BEARISH

    def test_empty_feed(self):
        news = build_news_sentiment("AAPL", {"feed": []})
        assert news.articles == []
        assert news.overall_sentiment_score is None
