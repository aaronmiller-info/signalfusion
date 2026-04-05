"""Base collector interface and signal database writer."""

from __future__ import annotations

import sqlite3
import time
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from signalfusion.data.schema import FEATURE_NAMES, NUM_FEATURES, SYMBOLS

DB_PATH = Path("data") / "signals.db"


def init_db(db_path: Path = DB_PATH) -> None:
    """Create the signals table if it doesn't exist."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    feature_cols = ", ".join(f'"{name}" REAL' for name in FEATURE_NAMES)
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS signals (
            symbol TEXT NOT NULL,
            timestamp REAL NOT NULL,
            {feature_cols},
            PRIMARY KEY (symbol, timestamp)
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_signals_symbol_ts
        ON signals (symbol, timestamp)
    """)
    conn.commit()
    conn.close()


def write_signals(
    symbol: str,
    timestamp: float,
    values: dict[str, float | None],
    db_path: Path = DB_PATH,
) -> None:
    """
    Upsert a single row of signals.

    Args:
        symbol: e.g. "BTC/USD"
        timestamp: Unix timestamp (seconds)
        values: dict mapping feature name → value (None for missing)
    """
    conn = sqlite3.connect(str(db_path))

    cols = ["symbol", "timestamp"] + FEATURE_NAMES
    placeholders = ", ".join(["?"] * len(cols))
    updates = ", ".join(f'"{name}" = excluded."{name}"' for name in FEATURE_NAMES)

    row = [symbol, timestamp] + [values.get(name) for name in FEATURE_NAMES]

    col_list = ", ".join(f'"{c}"' for c in cols)
    conn.execute(
        f"INSERT INTO signals ({col_list}) "
        f"VALUES ({placeholders}) "
        f"ON CONFLICT(symbol, timestamp) DO UPDATE SET {updates}",
        row,
    )
    conn.commit()
    conn.close()


def write_signals_batch(
    symbol: str,
    rows: list[tuple[float, dict[str, float | None]]],
    db_path: Path = DB_PATH,
) -> int:
    """
    Batch upsert multiple rows. Returns count of rows written.

    Args:
        symbol: e.g. "BTC/USD"
        rows: list of (timestamp, values_dict) tuples
    """
    if not rows:
        return 0

    conn = sqlite3.connect(str(db_path))
    cols = ["symbol", "timestamp"] + FEATURE_NAMES
    placeholders = ", ".join(["?"] * len(cols))
    updates = ", ".join(f'"{name}" = excluded."{name}"' for name in FEATURE_NAMES)

    batch = []
    for ts, values in rows:
        row = [symbol, ts] + [values.get(name) for name in FEATURE_NAMES]
        batch.append(row)

    col_list = ", ".join(f'"{c}"' for c in cols)
    conn.executemany(
        f"INSERT INTO signals ({col_list}) "
        f"VALUES ({placeholders}) "
        f"ON CONFLICT(symbol, timestamp) DO UPDATE SET {updates}",
        batch,
    )
    conn.commit()
    count = len(batch)
    conn.close()
    return count


class Collector(ABC):
    """Base class for signal collectors."""

    @abstractmethod
    async def collect(self, symbols: list[str]) -> dict[str, list[tuple[float, dict]]]:
        """
        Fetch signal data for the given symbols.

        Returns:
            dict mapping symbol → list of (timestamp, {feature_name: value}) tuples.
            Only include features this collector is responsible for.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Collector name for logging."""
        ...

    @property
    @abstractmethod
    def frequency_seconds(self) -> int:
        """How often this collector should run."""
        ...
