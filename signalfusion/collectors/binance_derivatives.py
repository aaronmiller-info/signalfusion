"""
Derivatives collector — fetches funding rates, open interest, and related
signals from OKX public API (US-accessible, no auth required).

Signals: funding_rate, open_interest, open_interest_delta, long_short_ratio,
         liquidation_volume, top_trader_long_ratio, basis_annualized, taker_buy_ratio

Note: Originally used Binance Futures API, switched to OKX because
Binance.com is geo-blocked in the US. OKX provides funding rate and
open interest publicly; other signals are stubbed as None.
"""

from __future__ import annotations

import time

import httpx

from signalfusion.collectors.base import Collector
from signalfusion.data.schema import SYMBOLS

# OKX instrument ID mapping (perpetual swaps)
OKX_INSTRUMENTS = {
    "BTC/USD": "BTC-USDT-SWAP",
    "ETH/USD": "ETH-USDT-SWAP",
    "SOL/USD": "SOL-USDT-SWAP",
    "XRP/USD": "XRP-USDT-SWAP",
    "LINK/USD": "LINK-USDT-SWAP",
    "AVAX/USD": "AVAX-USDT-SWAP",
}

OKX_BASE = "https://www.okx.com/api/v5"


class BinanceDerivativesCollector(Collector):
    """
    Fetches derivatives data from OKX public API.

    Class name kept as BinanceDerivativesCollector for backward compatibility
    with run.py imports.
    """

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
                inst_id = OKX_INSTRUMENTS.get(symbol)
                if not inst_id:
                    continue

                values: dict[str, float | None] = {}

                try:
                    # Funding rate
                    resp = await client.get(
                        f"{OKX_BASE}/public/funding-rate",
                        params={"instId": inst_id},
                    )
                    data = resp.json()
                    if data.get("code") == "0" and data.get("data"):
                        entry = data["data"][0]
                        values["funding_rate"] = float(entry.get("fundingRate", 0))

                    # Open interest (in USD)
                    resp = await client.get(
                        f"{OKX_BASE}/public/open-interest",
                        params={"instId": inst_id},
                    )
                    data = resp.json()
                    if data.get("code") == "0" and data.get("data"):
                        entry = data["data"][0]
                        values["open_interest"] = float(entry.get("oiUsd", 0))

                    # Signals not available from OKX public API without auth
                    values["open_interest_delta"] = None
                    values["long_short_ratio"] = None
                    values["liquidation_volume"] = None
                    values["top_trader_long_ratio"] = None
                    values["basis_annualized"] = None
                    values["taker_buy_ratio"] = None

                    result[symbol] = [(now, values)]

                except Exception as e:
                    print(f"  OKX derivatives error for {symbol}: {e}")

        return result
