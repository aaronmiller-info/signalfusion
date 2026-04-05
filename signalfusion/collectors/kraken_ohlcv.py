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

# Binance pair mapping (USDT pairs for deep history backfill)
BINANCE_PAIRS = {
    "BTC/USD": "BTCUSDT",
    "ETH/USD": "ETHUSDT",
    "SOL/USD": "SOLUSDT",
    "XRP/USD": "XRPUSDT",
    "LINK/USD": "LINKUSDT",
    "AVAX/USD": "AVAXUSDT",
}

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
        Fetch historical OHLCV from Binance klines API (deep history).

        Uses Binance USDT pairs for backfill since Kraken's OHLC endpoint
        is limited to ~720 recent bars. Price action is equivalent for
        training purposes.

        Args:
            symbol: trading pair (e.g. "BTC/USD")
            since: Unix timestamp to start from (None = auto-calculate from days)
            interval: bar interval in minutes (only 15 supported for backfill)
            days: number of days of history to fetch (default 90)

        Returns:
            List of (timestamp, values_dict) tuples
        """
        binance_pair = BINANCE_PAIRS.get(symbol)
        if not binance_pair:
            return []

        if since is None:
            since = int(time.time()) - (days * 86400)

        start_ms = since * 1000
        end_ms = int(time.time() * 1000)
        all_rows = []

        async with httpx.AsyncClient(timeout=30) as client:
            cursor_ms = start_ms
            while cursor_ms < end_ms:
                params = {
                    "symbol": binance_pair,
                    "interval": "15m",
                    "startTime": cursor_ms,
                    "limit": 1000,
                }
                resp = await client.get(
                    "https://api.binance.us/api/v3/klines", params=params
                )
                bars = resp.json()

                if not isinstance(bars, list) or len(bars) == 0:
                    break

                for bar in bars:
                    # Binance kline: [openTime, open, high, low, close, volume,
                    #                  closeTime, quoteVolume, trades, ...]
                    ts = bar[0] / 1000  # ms → seconds
                    values = {
                        "open": float(bar[1]),
                        "high": float(bar[2]),
                        "low": float(bar[3]),
                        "close": float(bar[4]),
                        "volume": float(bar[5]),
                        "trade_count": float(bar[8]),
                    }
                    all_rows.append((ts, values))

                # Advance cursor past the last bar
                cursor_ms = bars[-1][0] + 1

                if len(bars) < 1000:
                    break

                await asyncio.sleep(0.2)

        return all_rows
