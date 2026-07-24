"""
Alpha Vantage news + sentiment client.

Alpha Vantage's free tier allows only ~25 requests/day, so this client is
built to be *optional and frugal*:
- Feature-flagged: if no ALPHAVANTAGE_API_KEY is set, `is_enabled` is False and
  callers skip it entirely (no error).
- Heavily cached: a long TTL (default 6h) means repeat questions about the same
  ticker cost zero requests.
- Fail-soft: any error (rate limit, network, bad payload) returns None rather
  than raising, so news is never a hard dependency of an answer.

The HTTP call is injected so tests never touch the network.
"""

from __future__ import annotations

import logging
import os
import threading
import urllib.request
import json
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional
from urllib.parse import urlencode

logger = logging.getLogger(__name__)

NEWS_CACHE_TTL = timedelta(hours=6)
_BASE_URL = "https://www.alphavantage.co/query"
_MAX_ARTICLES = 5


def _default_http_get(url: str) -> dict:
    """Real HTTP GET returning parsed JSON. Isolated so it can be faked."""
    with urllib.request.urlopen(url, timeout=15) as resp:
        return json.loads(resp.read().decode("utf-8"))


class AlphaVantageNewsClient:
    """Optional, cached news-sentiment provider."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        http_get: Callable[[str], dict] = _default_http_get,
        cache_ttl: timedelta = NEWS_CACHE_TTL,
    ):
        self.api_key = api_key if api_key is not None else os.getenv("ALPHAVANTAGE_API_KEY")
        self._http_get = http_get
        self._ttl = cache_ttl
        self._cache: dict[str, tuple[datetime, Optional[dict]]] = {}
        self._lock = threading.Lock()

    @property
    def is_enabled(self) -> bool:
        """False when no API key is configured — callers should skip news."""
        return bool(self.api_key)

    def get_news_sentiment(self, ticker: str) -> Optional[dict]:
        """Return the raw Alpha Vantage news payload for a ticker, or None if
        unavailable (disabled, rate-limited, or errored). Parsing into models
        happens in the agent layer."""
        if not self.is_enabled:
            return None
        ticker = ticker.upper()
        now = datetime.now(timezone.utc)

        with self._lock:
            hit = self._cache.get(ticker)
            if hit and now - hit[0] < self._ttl:
                return hit[1]

        url = f"{_BASE_URL}?" + urlencode({
            "function": "NEWS_SENTIMENT",
            "tickers": ticker,
            "limit": _MAX_ARTICLES,
            "apikey": self.api_key,
        })
        try:
            payload = self._http_get(url)
        except Exception:
            logger.warning("Alpha Vantage news fetch failed for %s", ticker, exc_info=True)
            return None

        # Rate-limit / error responses come back as 200 with a Note/Information key.
        if not isinstance(payload, dict) or "feed" not in payload:
            note = (payload or {}).get("Note") or (payload or {}).get("Information")
            if note:
                logger.info("Alpha Vantage limited/unavailable: %s", note)
            payload = None

        with self._lock:
            self._cache[ticker] = (now, payload)
        return payload
