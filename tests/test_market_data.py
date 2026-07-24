"""Tests for the market data service: caching, TTL, failure isolation."""

from datetime import timedelta

from src.data.market_data import MarketDataService, QuoteResult

from tests.conftest import NOW, FakeFetcher, make_quote


class TestCaching:
    def test_second_call_served_from_cache(self, market_data, fake_fetcher):
        market_data.get_quotes(["AAPL"])
        market_data.get_quotes(["AAPL"])
        assert len(fake_fetcher.calls) == 1

    def test_only_uncached_tickers_fetched(self, market_data, fake_fetcher):
        market_data.get_quotes(["AAPL"])
        market_data.get_quotes(["AAPL", "MSFT"])
        assert fake_fetcher.calls == [["AAPL"], ["MSFT"]]

    def test_expired_cache_refetches(self, fake_fetcher):
        # Quotes are stamped NOW (fixed past time); zero TTL means always stale.
        svc = MarketDataService(fetcher=fake_fetcher, cache_ttl=timedelta(0))
        svc.get_quotes(["AAPL"])
        svc.get_quotes(["AAPL"])
        assert len(fake_fetcher.calls) == 2

    def test_input_tickers_deduped_and_uppercased(self, market_data, fake_fetcher):
        result = market_data.get_quotes(["aapl", "AAPL", "msft"])
        assert fake_fetcher.calls == [["AAPL", "MSFT"]]
        assert set(result.quotes) == {"AAPL", "MSFT"}

    def test_clear_cache(self, market_data, fake_fetcher):
        market_data.get_quotes(["AAPL"])
        market_data.clear_cache()
        market_data.get_quotes(["AAPL"])
        assert len(fake_fetcher.calls) == 2


class TestFailureHandling:
    def test_unknown_ticker_reported_not_fatal(self, market_data):
        result = market_data.get_quotes(["AAPL", "FAKETICK"])
        assert "AAPL" in result.quotes
        assert result.failed == ["FAKETICK"]

    def test_stale_quote_served_when_refetch_fails(self):
        fetcher = FakeFetcher({"AAPL": make_quote("AAPL", 200.0)})
        svc = MarketDataService(fetcher=fetcher, cache_ttl=timedelta(0))
        first = svc.get_quotes(["AAPL"])
        assert first.quotes["AAPL"].price == 200.0

        del fetcher.quotes["AAPL"]  # simulate API outage on refetch
        second = svc.get_quotes(["AAPL"])
        assert second.quotes["AAPL"].price == 200.0  # stale beats nothing
        assert second.failed == []

    def test_all_failures_returns_empty_quotes(self, market_data):
        result = market_data.get_quotes(["NOPE1", "NOPE2"])
        assert result.quotes == {}
        assert result.failed == ["NOPE1", "NOPE2"]
