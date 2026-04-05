"""
Sentiment collector — fetches market sentiment signals from free public APIs.
No API key required.

Signals:
  - fear_greed_index: Crypto Fear & Greed Index [0-100] (RAW norm)
  - reddit_mentions: Reddit mention count for symbol (ZSCORE norm) — stubbed None
  - reddit_sentiment: Reddit sentiment score [-1, 1] (RAW norm) — stubbed None
  - twitter_mention_velocity: Rate of change in Twitter/X mentions (ZSCORE norm) — stubbed None
  - google_trends: Google search interest [0-100] (ZSCORE norm) — stubbed None
  - news_sentiment: NLP-scored headline sentiment [-1, 1] (RAW norm) — stubbed None

Data sources:
  - Alternative.me Fear & Greed API (no auth): fear_greed_index
  - Reddit, Twitter/X, Google Trends, news NLP: require authenticated APIs — stubbed as None
"""

from __future__ import annotations

import time

import httpx

from signalfusion.collectors.base import Collector
from signalfusion.data.schema import SYMBOLS

FEAR_GREED_URL = "https://api.alternative.me/fng/?limit=1"


class SentimentCollector(Collector):
    """
    Fetches sentiment signals and broadcasts them to all symbols.

    fear_greed_index is a market-wide metric — every symbol receives the same
    value. All other sentiment signals require authenticated third-party APIs
    (Reddit, Twitter/X, Google Trends, news NLP) and are returned as None
    until those integrations are available.
    """

    @property
    def name(self) -> str:
        return "sentiment"

    @property
    def frequency_seconds(self) -> int:
        return 86400  # daily — fear & greed updates once per day

    async def collect(self, symbols: list[str] | None = None) -> dict[str, list[tuple[float, dict]]]:
        """
        Fetch sentiment signals and broadcast them to all symbols.

        Returns:
            dict mapping symbol → [(timestamp, {signal_name: value})]
            Each symbol receives one data point with the same sentiment values.
        """
        if symbols is None:
            symbols = SYMBOLS

        sentiment_values = await self._fetch_sentiment_values()
        if not sentiment_values:
            return {}

        timestamp = sentiment_values.pop("_timestamp", time.time())
        result: dict[str, list[tuple[float, dict]]] = {}
        for symbol in symbols:
            result[symbol] = [(timestamp, dict(sentiment_values))]

        return result

    async def _fetch_sentiment_values(self) -> dict[str, float | None]:
        """
        Fetch the current Fear & Greed Index value from Alternative.me.

        Returns a flat dict of signal_name → value, plus "_timestamp" for
        the data point's unix timestamp. Returns an empty dict on failure.

        Signals requiring authenticated APIs are set to None:
          - reddit_mentions
          - reddit_sentiment
          - twitter_mention_velocity
          - google_trends
          - news_sentiment
        """
        fear_greed_value, timestamp = await self._fetch_fear_greed()

        if fear_greed_value is None:
            # If the only live source fails, return empty so no stale row is written
            return {}

        return {
            "_timestamp": timestamp if timestamp is not None else time.time(),
            "fear_greed_index": fear_greed_value,
            # Require Reddit API (auth) — stubbed until integration is available
            "reddit_mentions": None,
            # Require Reddit API (auth) — stubbed until integration is available
            "reddit_sentiment": None,
            # Require Twitter/X API (auth) — stubbed until integration is available
            "twitter_mention_velocity": None,
            # Require Google Trends API (auth) — stubbed until integration is available
            "google_trends": None,
            # Require news NLP pipeline (auth) — stubbed until integration is available
            "news_sentiment": None,
        }

    async def _fetch_fear_greed(self) -> tuple[float | None, float | None]:
        """
        Fetch the latest Crypto Fear & Greed Index from Alternative.me.

        Returns:
            (value, timestamp) — both None on any error.
            value is a float in [0, 100].
            timestamp is a Unix timestamp (float seconds).
        """
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(FEAR_GREED_URL)
                resp.raise_for_status()
                data = resp.json()

                entries = data.get("data")
                if not entries:
                    print("  SentimentCollector: fear & greed API returned no data")
                    return None, None

                entry = entries[0]
                raw_value = entry.get("value")
                raw_timestamp = entry.get("timestamp")

                if raw_value is None:
                    print("  SentimentCollector: fear & greed entry missing 'value'")
                    return None, None

                value = float(raw_value)
                timestamp = float(raw_timestamp) if raw_timestamp is not None else None
                return value, timestamp

        except httpx.HTTPStatusError as e:
            print(f"  SentimentCollector: HTTP {e.response.status_code} fetching fear & greed")
            return None, None
        except Exception as e:
            print(f"  SentimentCollector: error fetching fear & greed: {e}")
            return None, None
