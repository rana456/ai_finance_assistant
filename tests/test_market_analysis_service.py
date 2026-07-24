"""Tests for MarketAnalysisService caching and AlphaVantageNewsClient."""

from datetime import timedelta

import pytest

from src.data.market_analysis_service import MarketAnalysisService
from src.data.news_client import AlphaVantageNewsClient

from tests.conftest import FakeQuoteFetcher, FakeHistoryFetcher


class TestQuoteCaching:
    def test_quote_served_from_cache(self, market_analysis_service, fake_quote_fetcher):
        market_analysis_service.get_quote("AAPL")
        market_analysis_service.get_quote("AAPL")
        assert fake_quote_fetcher.calls == ["AAPL"]

    def test_expired_quote_refetched(self, fake_quote_fetcher, fake_history_fetcher):
        svc = MarketAnalysisService(
            quote_fetcher=fake_quote_fetcher, history_fetcher=fake_history_fetcher,
            quote_ttl=timedelta(0),
        )
        svc.get_quote("AAPL")
        svc.get_quote("AAPL")
        assert len(fake_quote_fetcher.calls) == 2

    def test_batch_reports_failures(self, market_analysis_service):
        quotes, failed = market_analysis_service.get_quotes(["AAPL", "NOPE"])
        assert "AAPL" in quotes
        assert failed == ["NOPE"]

    def test_history_cached(self, market_analysis_service, fake_history_fetcher):
        market_analysis_service.get_history("AAPL", "1y")
        market_analysis_service.get_history("AAPL", "1y")
        assert fake_history_fetcher.calls == [("AAPL", "1y")]

    def test_unknown_ticker_returns_none(self, market_analysis_service):
        assert market_analysis_service.get_quote("NOPE") is None


class TestNewsClient:
    def test_disabled_without_key(self):
        client = AlphaVantageNewsClient(api_key="", http_get=lambda url: {})
        assert not client.is_enabled
        assert client.get_news_sentiment("AAPL") is None

    def test_enabled_with_key_returns_feed(self):
        payload = {"feed": [{"title": "x", "ticker_sentiment": []}]}
        calls = []
        def http(url):
            calls.append(url)
            return payload
        client = AlphaVantageNewsClient(api_key="KEY", http_get=http)
        assert client.is_enabled
        assert client.get_news_sentiment("AAPL") == payload
        # cached: second call doesn't re-fetch
        client.get_news_sentiment("AAPL")
        assert len(calls) == 1

    def test_rate_limit_note_becomes_none(self):
        note = {"Note": "Thank you for using Alpha Vantage! Our standard API rate limit is 25 requests per day."}
        client = AlphaVantageNewsClient(api_key="KEY", http_get=lambda url: note)
        assert client.get_news_sentiment("AAPL") is None

    def test_http_error_fails_soft(self):
        def boom(url):
            raise RuntimeError("network down")
        client = AlphaVantageNewsClient(api_key="KEY", http_get=boom)
        assert client.get_news_sentiment("AAPL") is None
