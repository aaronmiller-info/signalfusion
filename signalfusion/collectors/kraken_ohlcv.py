"""
Kraken OHLCV collector — fetches price bars from Kraken public API.
No API key required.

Signals: open, high, low, close, volume, trade_count
"""

from __future__ import annotations

import asyncio
import time

import httpx

from signalfusion.collectors.base import Collector
from signalfusion.data.schema import SYMBOLS

# Kraken pair name mapping
KRAKEN_PAIRS = {
    "BTC/USD": "XXBTZUSD",
    "ETH/USD": "XETHZUSD",
    "SOL/USD": "SOLUSD",
    "XRP/USD": "XXRPZUSD",
    "LINK/USD": "LINKUSD",
    "AVAX/USD": "AVAXUSD",
}


class KrakenOHLCVCollector(Collector):
    """Fetches OHLCV data from Kraken public REST API."""

    BASE_URL = "https://api.kraken.com/0/public"

    @property
    def name(self) -> str:
        return "kraken_ohlcv"

    @property
    def frequency_seconds(self) -> int:
        return 60  # every minute

    async def collect(self, symbols: list[str] | None = None) -> dict[str, list[tuple[float, dict]]]:
        if symbols is None:
            symbols = SYMBOLS

        result = {}
        async with httpx.AsyncClient(timeout=30) as client:
            for symbol in symbols:
                pair = KRAKEN_PAIRS.get(symbol)
                if not pair:
                    continue

                try:
                    resp = await client.get(
                        f"{self.BASE_URL}/OHLC",
                        params={"pair": pair, "interval": 1},  # 1-minute bars
                    )
                    data = resp.json()

                    if data.get("error"):
                        print(f"  Kraken error for {symbol}: {data['error']}")
                        continue

                    bars = list(data.get("result", {}).values())
                    if not bars or not isinstance(bars[0], list):
                        continue

                    rows = []
                    for bar in bars[0]:
                        # Kraken OHLC format: [time, open, high, low, close, vwap, volume, count]
                        ts = float(bar[0])
                        values = {
                            "open": float(bar[1]),
                            "high": float(bar[2]),
                            "low": float(bar[3]),
                            "close": float(bar[4]),
                            "volume": float(bar[6]),
                            "trade_count": float(bar[7]),
                        }
                        rows.append((ts, values))

                    result[symbol] = rows

                except Exception as e:
                    print(f"  Kraken OHLCV error for {symbol}: {e}")

        return result

    async def backfill(
        self,
        symbol: str = "BTC/USD",
        since: int | None = None,
        interval: int = 15,
        days: int = 90,
    ) -> list[tuple[float, dict]]:
        """
        Fetch historical OHLCV from Kraken with pagination.

        Args:
            symbol: trading pair
            since: Unix timestamp to start from (None = auto-calculate from days)
            interval: bar interval in minutes (1, 5, 15, 30, 60, 240, 1440)
            days: number of days of history to fetch (default 90)

        Returns:
            List of (timestamp, values_dict) tuples
        """
        pair = KRAKEN_PAIRS.get(symbol)
        if not pair:
            return []

        if since is None:
            since = int(time.time()) - (days * 86400)

        all_rows = []
        cursor = since

        async with httpx.AsyncClient(timeout=30) as client:
            while True:
                params = {"pair": pair, "interval": interval, "since": cursor}
                resp = await client.get(f"{self.BASE_URL}/OHLC", params=params)
                data = resp.json()

                if data.get("error"):
                    print(f"Kraken backfill error: {data['error']}")
                    break

                # Get the "last" cursor for pagination
                result = data.get("result", {})
                last = result.pop("last", None)

                bars = list(result.values())
                if not bars or not isinstance(bars[0], list) or len(bars[0]) == 0:
                    break

                batch_count = 0
                for bar in bars[0]:
                    ts = float(bar[0])
                    values = {
                        "open": float(bar[1]),
                        "high": float(bar[2]),
                        "low": float(bar[3]),
                        "close": float(bar[4]),
                        "volume": float(bar[6]),
                        "trade_count": float(bar[7]),
                    }
                    all_rows.append((ts, values))
                    batch_count += 1

                # If we got fewer than 720 bars, we've reached the end
                if batch_count < 700 or last is None:
                    break

                cursor = last
                # Small delay to be respectful to the API
                await asyncio.sleep(1)

        # Deduplicate by timestamp and sort
        seen = {}
        for ts, vals in all_rows:
            seen[ts] = vals
        all_rows = sorted(seen.items(), key=lambda x: x[0])

        return all_rows
