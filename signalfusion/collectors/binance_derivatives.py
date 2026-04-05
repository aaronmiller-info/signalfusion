"""
Binance derivatives collector — fetches funding rates, open interest,
long/short ratios, and liquidations from Binance Futures public API.
No API key required for public endpoints.

Signals: funding_rate, open_interest, open_interest_delta, long_short_ratio,
         liquidation_volume, top_trader_long_ratio, basis_annualized, taker_buy_ratio
"""

from __future__ import annotations

import time

import httpx

from signalfusion.collectors.base import Collector
from signalfusion.data.schema import SYMBOLS

# Binance futures symbol mapping
BINANCE_SYMBOLS = {
    "BTC/USD": "BTCUSDT",
    "ETH/USD": "ETHUSDT",
    "SOL/USD": "SOLUSDT",
    "XRP/USD": "XRPUSDT",
    "LINK/USD": "LINKUSDT",
    "AVAX/USD": "AVAXUSDT",
}


class BinanceDerivativesCollector(Collector):
    """Fetches derivatives data from Binance Futures public API."""

    BASE_URL = "https://fapi.binance.com"

    @property
    def name(self) -> str:
        return "binance_derivatives"

    @property
    def frequency_seconds(self) -> int:
        return 300  # every 5 minutes

    async def collect(self, symbols: list[str] | None = None) -> dict[str, list[tuple[float, dict]]]:
        if symbols is None:
            symbols = SYMBOLS

        result = {}
        now = time.time()

        async with httpx.AsyncClient(timeout=30) as client:
            for symbol in symbols:
                bsym = BINANCE_SYMBOLS.get(symbol)
                if not bsym:
                    continue

                values: dict[str, float | None] = {}

                try:
                    # Funding rate
                    resp = await client.get(
                        f"{self.BASE_URL}/fapi/v1/fundingRate",
                        params={"symbol": bsym, "limit": 1},
                    )
                    data = resp.json()
                    if data and isinstance(data, list):
                        values["funding_rate"] = float(data[-1].get("fundingRate", 0))

                    # Open interest
                    resp = await client.get(
                        f"{self.BASE_URL}/fapi/v1/openInterest",
                        params={"symbol": bsym},
                    )
                    data = resp.json()
                    if isinstance(data, dict):
                        values["open_interest"] = float(data.get("openInterest", 0))

                    # Long/short ratio (global)
                    resp = await client.get(
                        f"{self.BASE_URL}/futures/data/globalLongShortAccountRatio",
                        params={"symbol": bsym, "period": "5m", "limit": 1},
                    )
                    data = resp.json()
                    if data and isinstance(data, list):
                        values["long_short_ratio"] = float(data[-1].get("longShortRatio", 1.0))

                    # Top trader long/short ratio
                    resp = await client.get(
                        f"{self.BASE_URL}/futures/data/topLongShortPositionRatio",
                        params={"symbol": bsym, "period": "5m", "limit": 1},
                    )
                    data = resp.json()
                    if data and isinstance(data, list):
                        values["top_trader_long_ratio"] = float(
                            data[-1].get("longAccount", 0.5)
                        )

                    # Taker buy/sell volume
                    resp = await client.get(
                        f"{self.BASE_URL}/futures/data/takerlongshortRatio",
                        params={"symbol": bsym, "period": "5m", "limit": 1},
                    )
                    data = resp.json()
                    if data and isinstance(data, list):
                        values["taker_buy_ratio"] = float(data[-1].get("buyVol", 0.5))

                    result[symbol] = [(now, values)]

                except Exception as e:
                    print(f"  Binance derivatives error for {symbol}: {e}")

        return result
