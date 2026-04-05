"""
Data preparation and loading for SignalFusion experiments.
DO NOT MODIFY — this file is read-only for autoresearch.

Loads signals from signals.db, normalizes per schema, creates
train/val/holdout splits with regime labeling, and provides
a DataLoader interface for the training loop.
"""

from __future__ import annotations

import math
import sqlite3
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from signalfusion.data.schema import (
    FEATURE_NAMES, NUM_FEATURES, SYMBOLS, RESAMPLE_MINUTES,
    SIGNAL_BY_NAME, NormMethod,
)

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

TIME_BUDGET = 300               # 5 minutes total for train + eval
DATA_DIR = Path("data")
DB_PATH = DATA_DIR / "signals.db"

# Split ratios (by time, not random)
TRAIN_RATIO = 0.55              # first 55% of data
VAL_RATIO = 0.20                # next 20%
HOLDOUT_RATIO = 0.25            # final 25% — never seen during autoresearch

# Regime detection (for anti-overfit constraint)
REGIME_WINDOW_BARS = 96 * 4     # ~4 days at 15min bars per regime window
NUM_REGIMES = 4                 # split val into 4 time windows for regime check

# Normalization
ZSCORE_LOOKBACK = 2880          # 30 days of 15-min bars for rolling z-score

# Target labels
FORWARD_BARS_DEFAULT = 4        # 1 hour at 15-min bars
THRESHOLD_PCT = 0.5             # ±0.5% for long/short classification


# ---------------------------------------------------------------------------
# Database loading
# ---------------------------------------------------------------------------

def load_signals_from_db(
    db_path: Path = DB_PATH,
    symbols: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """
    Load all signals from SQLite into numpy arrays.

    Returns:
        dict mapping symbol → array of shape (num_bars, num_features).
        Bars are sorted by timestamp ascending. Features are in FEATURE_NAMES order.
        Missing values are NaN.
    """
    if symbols is None:
        symbols = SYMBOLS

    conn = sqlite3.connect(str(db_path))
    result = {}

    for symbol in symbols:
        query = f"""
            SELECT {', '.join(FEATURE_NAMES)}
            FROM signals
            WHERE symbol = ?
            ORDER BY timestamp ASC
        """
        cursor = conn.execute(query, (symbol,))
        rows = cursor.fetchall()
        if rows:
            result[symbol] = np.array(rows, dtype=np.float64)
        else:
            result[symbol] = np.empty((0, NUM_FEATURES), dtype=np.float64)

    conn.close()
    return result


def load_timestamps_from_db(
    db_path: Path = DB_PATH,
    symbol: str = "BTC/USD",
) -> np.ndarray:
    """Load timestamps for splitting."""
    conn = sqlite3.connect(str(db_path))
    cursor = conn.execute(
        "SELECT timestamp FROM signals WHERE symbol = ? ORDER BY timestamp ASC",
        (symbol,),
    )
    ts = np.array([row[0] for row in cursor.fetchall()], dtype=np.float64)
    conn.close()
    return ts


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize_features(data: np.ndarray, lookback: int = ZSCORE_LOOKBACK) -> np.ndarray:
    """
    Apply per-feature normalization according to schema.

    Args:
        data: shape (num_bars, num_features), raw values
        lookback: rolling window for z-score normalization

    Returns:
        Normalized array, same shape. NaN where insufficient history.
    """
    out = np.full_like(data, np.nan)
    num_bars, num_feats = data.shape

    for i, name in enumerate(FEATURE_NAMES):
        sig = SIGNAL_BY_NAME[name]
        col = data[:, i]

        if sig.norm == NormMethod.LOG_RETURN:
            # log(x_t / x_{t-1}), first value is NaN
            safe = np.where(col > 0, col, np.nan)
            out[1:, i] = np.log(safe[1:] / safe[:-1])

        elif sig.norm == NormMethod.DIFF:
            out[1:, i] = col[1:] - col[:-1]

        elif sig.norm == NormMethod.RAW:
            # Already bounded (ratios, indices), just scale to ~[-1, 1]
            col_min = np.nanmin(col)
            col_max = np.nanmax(col)
            if col_max > col_min:
                out[:, i] = 2 * (col - col_min) / (col_max - col_min) - 1
            else:
                out[:, i] = 0.0

        elif sig.norm == NormMethod.RANK:
            # Percentile rank in rolling window → [0, 1]
            for t in range(lookback, num_bars):
                window = col[t - lookback : t]
                valid = window[~np.isnan(window)]
                if len(valid) > 1:
                    out[t, i] = np.searchsorted(np.sort(valid), col[t]) / len(valid)

        else:  # ZSCORE (default)
            # Rolling z-score
            for t in range(lookback, num_bars):
                window = col[t - lookback : t]
                valid = window[~np.isnan(window)]
                if len(valid) > 10:
                    mu = np.mean(valid)
                    sigma = np.std(valid)
                    if sigma > 1e-10:
                        out[t, i] = (col[t] - mu) / sigma
                    else:
                        out[t, i] = 0.0

    return out


# ---------------------------------------------------------------------------
# Target labels
# ---------------------------------------------------------------------------

def compute_labels(
    close_prices: np.ndarray,
    forward_bars: int = FORWARD_BARS_DEFAULT,
    threshold_pct: float = THRESHOLD_PCT,
) -> np.ndarray:
    """
    Compute classification labels based on forward returns.

    Returns:
        Array of shape (num_bars,) with values:
        0 = short (return < -threshold)
        1 = flat  (-threshold <= return <= +threshold)
        2 = long  (return > +threshold)
        -1 = no label (insufficient forward data)
    """
    num_bars = len(close_prices)
    labels = np.full(num_bars, -1, dtype=np.int64)

    for t in range(num_bars - forward_bars):
        ret = (close_prices[t + forward_bars] - close_prices[t]) / close_prices[t] * 100
        if ret < -threshold_pct:
            labels[t] = 0  # short
        elif ret > threshold_pct:
            labels[t] = 2  # long
        else:
            labels[t] = 1  # flat

    return labels


def compute_forward_returns(
    close_prices: np.ndarray,
    forward_bars: int = FORWARD_BARS_DEFAULT,
) -> np.ndarray:
    """Compute raw forward returns (for Sharpe-based loss)."""
    num_bars = len(close_prices)
    returns = np.full(num_bars, np.nan, dtype=np.float64)
    for t in range(num_bars - forward_bars):
        returns[t] = (close_prices[t + forward_bars] - close_prices[t]) / close_prices[t]
    return returns


# ---------------------------------------------------------------------------
# Regime detection (simple volatility-based)
# ---------------------------------------------------------------------------

def label_regimes(close_prices: np.ndarray, window: int = REGIME_WINDOW_BARS) -> np.ndarray:
    """
    Label each bar with a regime based on rolling volatility and trend.

    0 = trending (strong directional move, moderate vol)
    1 = ranging  (low vol, no trend)
    2 = volatile (high vol, no clear trend)
    3 = quiet    (very low vol)

    Used to split validation into regime windows for anti-overfit checking.
    """
    num_bars = len(close_prices)
    regimes = np.zeros(num_bars, dtype=np.int64)

    for t in range(window, num_bars):
        w = close_prices[t - window : t]
        returns = np.diff(np.log(np.maximum(w, 1e-10)))
        vol = np.std(returns)
        trend = abs(np.mean(returns)) / max(vol, 1e-10)

        vol_median = 0.01  # rough threshold
        if vol > vol_median * 1.5:
            regimes[t] = 2  # volatile
        elif vol < vol_median * 0.5:
            regimes[t] = 3  # quiet
        elif trend > 0.3:
            regimes[t] = 0  # trending
        else:
            regimes[t] = 1  # ranging

    return regimes


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SignalDataset(Dataset):
    """
    Windowed dataset: each sample is a (features, label, forward_return) tuple.

    features: shape (seq_len, num_features) — normalized signal values
    label: int in {0, 1, 2} — short/flat/long classification
    forward_return: float — raw forward return for Sharpe-based training
    """

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        forward_returns: np.ndarray,
        seq_len: int = 200,
    ):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.forward_returns = torch.tensor(forward_returns, dtype=torch.float32)
        self.seq_len = seq_len

        # Build valid indices (have full window AND valid label)
        self.valid_indices = []
        for t in range(seq_len, len(features)):
            if labels[t] >= 0:
                self.valid_indices.append(t)
        self.valid_indices = np.array(self.valid_indices)

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t = self.valid_indices[idx]
        x = self.features[t - self.seq_len : t]  # (seq_len, num_features)
        y = self.labels[t]
        r = self.forward_returns[t]
        return x, y, r


# ---------------------------------------------------------------------------
# Data preparation pipeline
# ---------------------------------------------------------------------------

def prepare_datasets(
    symbol: str = "BTC/USD",
    seq_len: int = 200,
    forward_bars: int = FORWARD_BARS_DEFAULT,
    threshold_pct: float = THRESHOLD_PCT,
) -> tuple[SignalDataset, SignalDataset, SignalDataset, dict]:
    """
    Full data preparation pipeline.

    Returns:
        (train_dataset, val_dataset, holdout_dataset, metadata)

    metadata includes:
        - num_bars: total bars loaded
        - train_end_idx, val_end_idx: split points
        - val_regime_windows: list of (start, end) tuples for regime-based eval
    """
    # Load raw data
    raw_data = load_signals_from_db()
    if symbol not in raw_data or len(raw_data[symbol]) == 0:
        raise ValueError(f"No data for {symbol} in {DB_PATH}")

    raw = raw_data[symbol]
    num_bars = len(raw)

    # Detect active features (columns that have at least some non-NaN data)
    active_mask = ~np.all(np.isnan(raw), axis=0)
    active_indices = np.where(active_mask)[0]
    active_names = [FEATURE_NAMES[i] for i in active_indices]
    print(f"  Active features: {len(active_names)}/{NUM_FEATURES} — {active_names}")

    # Normalize (full array)
    normalized = normalize_features(raw)

    # Keep only active features and fill remaining NaN with 0
    normalized = normalized[:, active_indices]
    normalized = np.nan_to_num(normalized, nan=0.0)

    # Close prices for labels (column index from schema)
    close_idx = FEATURE_NAMES.index("close")
    close_prices = raw[:, close_idx]

    # Labels and forward returns
    labels = compute_labels(close_prices, forward_bars, threshold_pct)
    forward_returns = compute_forward_returns(close_prices, forward_bars)

    # Time-based splits
    train_end = int(num_bars * TRAIN_RATIO)
    val_end = int(num_bars * (TRAIN_RATIO + VAL_RATIO))

    train_ds = SignalDataset(
        normalized[:train_end], labels[:train_end],
        forward_returns[:train_end], seq_len,
    )
    val_ds = SignalDataset(
        normalized[train_end:val_end], labels[train_end:val_end],
        forward_returns[train_end:val_end], seq_len,
    )
    holdout_ds = SignalDataset(
        normalized[val_end:], labels[val_end:],
        forward_returns[val_end:], seq_len,
    )

    # Regime windows for val set
    val_bars = val_end - train_end
    window_size = val_bars // NUM_REGIMES
    val_regime_windows = [
        (i * window_size, min((i + 1) * window_size, val_bars))
        for i in range(NUM_REGIMES)
    ]

    metadata = {
        "num_bars": num_bars,
        "train_bars": train_end,
        "val_bars": val_bars,
        "holdout_bars": num_bars - val_end,
        "train_end_idx": train_end,
        "val_end_idx": val_end,
        "val_regime_windows": val_regime_windows,
        "close_prices_val": close_prices[train_end:val_end],
        "active_features": active_names,
        "num_active_features": len(active_names),
    }

    return train_ds, val_ds, holdout_ds, metadata


def make_dataloader(
    dataset: SignalDataset,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Create a DataLoader from a SignalDataset."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
