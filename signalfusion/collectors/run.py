"""
Collector runner — orchestrates all data collectors.

Usage:
    python -m signalfusion.collectors.run                  # one-shot fetch
    python -m signalfusion.collectors.run --daemon          # continuous collection
    python -m signalfusion.collectors.run --backfill        # historical backfill
"""

from __future__ import annotations

import argparse
import asyncio
import time

from signalfusion.collectors.base import init_db, write_signals_batch, DB_PATH
from signalfusion.collectors.kraken_ohlcv import KrakenOHLCVCollector
from signalfusion.collectors.binance_derivatives import BinanceDerivativesCollector
from signalfusion.data.schema import SYMBOLS


ALL_COLLECTORS = [
    KrakenOHLCVCollector(),
    BinanceDerivativesCollector(),
]


async def run_once() -> None:
    """Run all collectors once."""
    print(f"Collecting signals for {len(SYMBOLS)} symbols...")

    for collector in ALL_COLLECTORS:
        print(f"  Running {collector.name}...")
        try:
            data = await collector.collect()
            total = 0
            for symbol, rows in data.items():
                count = write_signals_batch(symbol, rows)
                total += count
            print(f"    → {total} rows written")
        except Exception as e:
            print(f"    → ERROR: {e}")


async def run_daemon() -> None:
    """Run collectors continuously on their configured schedules."""
    print("Starting signal collection daemon...")
    print(f"Database: {DB_PATH}")

    while True:
        for collector in ALL_COLLECTORS:
            try:
                data = await collector.collect()
                for symbol, rows in data.items():
                    write_signals_batch(symbol, rows)
            except Exception as e:
                print(f"  {collector.name} error: {e}")

        # Sleep until next collection cycle (shortest frequency)
        min_freq = min(c.frequency_seconds for c in ALL_COLLECTORS)
        await asyncio.sleep(min_freq)


async def run_backfill() -> None:
    """Backfill historical data from available sources."""
    print("Backfilling historical data...")

    kraken = KrakenOHLCVCollector()
    for symbol in SYMBOLS:
        print(f"  Backfilling {symbol} from Kraken (15-min bars)...")
        try:
            rows = await kraken.backfill(symbol, interval=15)
            count = write_signals_batch(symbol, rows)
            print(f"    → {count} bars written")
        except Exception as e:
            print(f"    → ERROR: {e}")


def main():
    parser = argparse.ArgumentParser(description="SignalFusion data collector")
    parser.add_argument("--daemon", action="store_true", help="Run continuously")
    parser.add_argument("--backfill", action="store_true", help="Historical backfill")
    args = parser.parse_args()

    init_db()

    if args.backfill:
        asyncio.run(run_backfill())
    elif args.daemon:
        asyncio.run(run_daemon())
    else:
        asyncio.run(run_once())


if __name__ == "__main__":
    main()
