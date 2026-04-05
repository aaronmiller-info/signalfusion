"""
CoinGecko collector — fetches market structure signals from CoinGecko's free API.
No API key required. Rate limit: 30 requests/minute.

Signals:
  - btc_dominance: BTC market cap as % of total crypto (DIFF norm)
  - eth_btc_ratio: ETH/BTC price ratio (LOG_RETURN norm)
  - stablecoin_market_cap: Total stablecoin market cap in USD (LOG_RETURN norm)
  - exchange_volume_hhi: Exchange volume concentration HHI (ZSCORE norm) — stubbed None
  - alt_btc_correlation: Rolling 30d correlation of top alts with BTC (RAW norm) — stubbed None

Data sources:
  - Global endpoint: https://api.coingecko.com/api/v3/global
      → market_cap_percentage.btc (btc_dominance)
  - Simple price endpoint: https://api.coingecko.com/api/v3/simple/price
      → ETH price in BTC (eth_btc_ratio)
  - Coins markets endpoint: https://api.coingecko.com/api/v3/coins/markets
      → stablecoin category market caps summed (stablecoin_market_cap)

All signals are market-wide and broadcast identically to every symbol in SYMBOLS.
"""

from __future__ import annotations

import asyncio
import time

import httpx

from signalfusion.collectors.base import Collector
from signalfusion.data.schema import SYMBOLS

# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

COINGECKO_BASE = "https://api.coingecko.com/api/v3"

GLOBAL_URL = f"{COINGECKO_BASE}/global"
SIMPLE_PRICE_URL = f"{COINGECKO_BASE}/simple/price"
COINS_MARKETS_URL = f"{COINGECKO_BASE}/coins/markets"

# Delay between sequential requests to stay well under 30 req/min
_REQUEST_DELAY_SECONDS = 2.5


class CoinGeckoCollector(Collector):
    """
    Fetches market structure signals from CoinGecko's free public API.

    All five signals are market-wide metrics — every symbol in SYMBOLS receives
    the same values stamped at the same timestamp.

    exchange_volume_hhi and alt_btc_correlation require either exchange-level
    volume data or a historical price series for correlation computation; both
    are returned as None until those data sources are available.
    """

    @property
    def name(self) -> str:
        return "coingecko"

    @property
    def frequency_seconds(self) -> int:
        return 3600  # hourly — btc_dominance and eth_btc_ratio update frequently enough

    async def collect(
        self, symbols: list[str] | None = None
    ) -> dict[str, list[tuple[float, dict]]]:
        """
        Fetch market structure signals and broadcast them to all symbols.

        Returns:
            dict mapping symbol → [(timestamp, {signal_name: value})]
            Each symbol receives one data point with the same market structure values.
        """
        if symbols is None:
            symbols = SYMBOLS

        market_values = await self._fetch_market_structure_values()
        if not market_values:
            return {}

        timestamp = market_values.pop("_timestamp", time.time())
        result: dict[str, list[tuple[float, dict]]] = {}
        for symbol in symbols:
            result[symbol] = [(timestamp, dict(market_values))]

        return result

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    async def _fetch_market_structure_values(self) -> dict[str, float | None]:
        """
        Fetch all market structure signals sequentially (respects rate limit).

        Returns a flat dict of signal_name → value, plus "_timestamp".
        Returns an empty dict if the primary global fetch fails.
        """
        try:
            async with httpx.AsyncClient(
                timeout=30,
                headers={"User-Agent": "signalfusion/1.0"},
            ) as client:
                # 1. Global — provides btc_dominance and a timestamp anchor
                btc_dominance, timestamp = await self._fetch_btc_dominance(client)
                if btc_dominance is None:
                    # Global endpoint is the anchor; abort if it fails
                    return {}

                await asyncio.sleep(_REQUEST_DELAY_SECONDS)

                # 2. Simple price — ETH/BTC ratio
                eth_btc_ratio = await self._fetch_eth_btc_ratio(client)

                await asyncio.sleep(_REQUEST_DELAY_SECONDS)

                # 3. Stablecoin market cap — sum top-10 stablecoins by market cap
                stablecoin_market_cap = await self._fetch_stablecoin_market_cap(client)

        except Exception as e:
            print(f"  CoinGeckoCollector: unexpected error during fetch: {e}")
            return {}

        return {
            "_timestamp": timestamp if timestamp is not None else time.time(),
            "btc_dominance": btc_dominance,
            "eth_btc_ratio": eth_btc_ratio,
            "stablecoin_market_cap": stablecoin_market_cap,
            # Requires per-exchange volume breakdown — not available on free tier
            "exchange_volume_hhi": None,
            # Requires a 30-day historical price series for each alt — not fetched here
            "alt_btc_correlation": None,
        }

    # ------------------------------------------------------------------
    # Individual fetchers
    # ------------------------------------------------------------------

    async def _fetch_btc_dominance(
        self, client: httpx.AsyncClient
    ) -> tuple[float | None, float | None]:
        """
        Fetch BTC dominance (%) from the /global endpoint.

        Returns:
            (btc_dominance_pct, server_updated_at_timestamp)
            Both None on any error.
        """
        try:
            resp = await client.get(GLOBAL_URL)
            resp.raise_for_status()
            data = resp.json()

            global_data = data.get("data", {})
            market_cap_pct = global_data.get("market_cap_percentage", {})
            btc_pct = market_cap_pct.get("btc")

            if btc_pct is None:
                print("  CoinGeckoCollector: 'market_cap_percentage.btc' missing from /global")
                return None, None

            # updated_at is a Unix timestamp returned by the API
            updated_at = global_data.get("updated_at")
            timestamp = float(updated_at) if updated_at is not None else None

            return float(btc_pct), timestamp

        except httpx.HTTPStatusError as e:
            print(f"  CoinGeckoCollector: HTTP {e.response.status_code} fetching /global")
            return None, None
        except Exception as e:
            print(f"  CoinGeckoCollector: error fetching /global: {e}")
            return None, None

    async def _fetch_eth_btc_ratio(
        self, client: httpx.AsyncClient
    ) -> float | None:
        """
        Fetch the ETH/BTC price ratio from the /simple/price endpoint.

        Returns:
            Float ratio (ETH price denominated in BTC), or None on any error.
        """
        params = {
            "ids": "ethereum",
            "vs_currencies": "btc",
        }
        try:
            resp = await client.get(SIMPLE_PRICE_URL, params=params)
            resp.raise_for_status()
            data = resp.json()

            eth_data = data.get("ethereum", {})
            eth_in_btc = eth_data.get("btc")

            if eth_in_btc is None:
                print("  CoinGeckoCollector: 'ethereum.btc' missing from /simple/price")
                return None

            return float(eth_in_btc)

        except httpx.HTTPStatusError as e:
            print(
                f"  CoinGeckoCollector: HTTP {e.response.status_code} fetching ETH/BTC ratio"
            )
            return None
        except Exception as e:
            print(f"  CoinGeckoCollector: error fetching ETH/BTC ratio: {e}")
            return None

    async def _fetch_stablecoin_market_cap(
        self, client: httpx.AsyncClient
    ) -> float | None:
        """
        Fetch and sum the market caps of the top 10 stablecoins by market cap.

        Uses the /coins/markets endpoint filtered by the 'stablecoins' category.

        Returns:
            Total stablecoin market cap in USD (float), or None on any error.
        """
        params = {
            "vs_currency": "usd",
            "category": "stablecoins",
            "order": "market_cap_desc",
            "per_page": 10,
            "page": 1,
            "sparkline": "false",
        }
        try:
            resp = await client.get(COINS_MARKETS_URL, params=params)
            resp.raise_for_status()
            coins = resp.json()

            if not coins:
                print("  CoinGeckoCollector: /coins/markets returned no stablecoin entries")
                return None

            total = 0.0
            counted = 0
            for coin in coins:
                mc = coin.get("market_cap")
                if mc is not None:
                    total += float(mc)
                    counted += 1

            if counted == 0:
                print("  CoinGeckoCollector: all stablecoin market_cap values are None")
                return None

            return total

        except httpx.HTTPStatusError as e:
            print(
                f"  CoinGeckoCollector: HTTP {e.response.status_code} fetching stablecoin market caps"
            )
            return None
        except Exception as e:
            print(f"  CoinGeckoCollector: error fetching stablecoin market caps: {e}")
            return None
