"""
Macro economic collector — fetches global macro signals from free public APIs.
No API key required.

Signals: dxy, us_10y_yield, us_2y10y_spread, vix, gold, sp500,
         global_m2 (None — requires FRED key), fed_funds_rate (None — requires FRED key)

Data sources:
  - Yahoo Finance chart API (no auth): DXY, 10Y yield, 2Y yield, VIX, Gold, S&P 500
  - FRED API (requires key): global_m2, fed_funds_rate — stubbed as None for now
"""

from __future__ import annotations

import time

import httpx

from signalfusion.collectors.base import Collector
from signalfusion.data.schema import SYMBOLS

# Yahoo Finance ticker → signal name mapping
# These are all daily signals that apply equally to every crypto symbol.
YAHOO_TICKERS: dict[str, str] = {
    "DX-Y.NYB": "dxy",        # US Dollar Index
    "^TNX": "us_10y_yield",   # 10-year Treasury yield
    "^IRX": "us_2y_yield",    # 2-year Treasury yield (used to compute spread)
    "^VIX": "vix",            # CBOE Volatility Index
    "GC=F": "gold",           # Gold front-month futures (XAU/USD proxy)
    "^GSPC": "sp500",         # S&P 500
}

YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"


class MacroCollector(Collector):
    """
    Fetches daily macro signals from Yahoo Finance public chart API.

    All signals are symbol-agnostic — every crypto symbol in SYMBOLS receives
    the same macro values stamped at the same daily timestamp.

    global_m2 and fed_funds_rate are returned as None pending a FRED API key.
    """

    @property
    def name(self) -> str:
        return "macro"

    @property
    def frequency_seconds(self) -> int:
        return 86400  # daily

    async def collect(self, symbols: list[str] | None = None) -> dict[str, list[tuple[float, dict]]]:
        """
        Fetch macro signals and broadcast them to all symbols.

        Returns:
            dict mapping symbol → [(timestamp, {signal_name: value})]
            Each symbol receives one data point with the same macro values.
        """
        if symbols is None:
            symbols = SYMBOLS

        macro_values = await self._fetch_macro_values()
        if not macro_values:
            return {}

        # Broadcast identical macro values to every symbol
        timestamp = macro_values.pop("_timestamp", time.time())
        result: dict[str, list[tuple[float, dict]]] = {}
        for symbol in symbols:
            result[symbol] = [(timestamp, dict(macro_values))]

        return result

    async def _fetch_macro_values(self) -> dict[str, float | None]:
        """
        Fetch today's closing values for all Yahoo Finance tickers.

        Returns a flat dict of signal_name → value, plus "_timestamp" for
        the data point's unix timestamp. Returns an empty dict on failure.
        """
        raw: dict[str, float | None] = {}

        try:
            async with httpx.AsyncClient(
                timeout=30,
                headers={"User-Agent": "Mozilla/5.0"},
            ) as client:
                for ticker, signal in YAHOO_TICKERS.items():
                    value, ts = await self._fetch_ticker(client, ticker)
                    raw[signal] = value
                    if ts is not None:
                        raw["_timestamp"] = ts
        except Exception as e:
            print(f"  MacroCollector: unexpected error during fetch: {e}")
            return {}

        # Derive 2Y-10Y spread from the individual yields
        yield_10y = raw.get("us_10y_yield")
        yield_2y = raw.get("us_2y_yield")
        if yield_10y is not None and yield_2y is not None:
            raw["us_2y10y_spread"] = yield_10y - yield_2y
        else:
            raw["us_2y10y_spread"] = None

        # Drop the intermediate 2Y yield — it is not a schema signal
        raw.pop("us_2y_yield", None)

        # Monthly FRED signals — stubbed until an API key is available
        raw["global_m2"] = None
        raw["fed_funds_rate"] = None

        return raw

    async def _fetch_ticker(
        self, client: httpx.AsyncClient, ticker: str
    ) -> tuple[float | None, float | None]:
        """
        Fetch the most recent closing price for a single Yahoo Finance ticker.

        Returns:
            (price, timestamp) — both None on any error.
        """
        url = YAHOO_CHART_URL.format(ticker=ticker)
        params = {"interval": "1d", "range": "1d"}

        try:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()

            chart = data.get("chart", {})
            if chart.get("error"):
                print(f"  MacroCollector: Yahoo error for {ticker}: {chart['error']}")
                return None, None

            results = chart.get("result")
            if not results:
                print(f"  MacroCollector: no results for {ticker}")
                return None, None

            result = results[0]
            timestamps: list[int] = result.get("timestamp", [])
            closes: list[float | None] = (
                result.get("indicators", {})
                .get("quote", [{}])[0]
                .get("close", [])
            )

            if not timestamps or not closes:
                print(f"  MacroCollector: empty series for {ticker}")
                return None, None

            # Walk backwards to find the last non-None close
            for i in range(len(closes) - 1, -1, -1):
                if closes[i] is not None:
                    return float(closes[i]), float(timestamps[i])

            print(f"  MacroCollector: all closes are None for {ticker}")
            return None, None

        except httpx.HTTPStatusError as e:
            print(f"  MacroCollector: HTTP {e.response.status_code} for {ticker}")
            return None, None
        except Exception as e:
            print(f"  MacroCollector: error fetching {ticker}: {e}")
            return None, None
