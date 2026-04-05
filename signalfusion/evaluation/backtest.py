"""
Backtest evaluation harness for SignalFusion experiments.
DO NOT MODIFY — this file is read-only for autoresearch.

Takes model predictions on the validation set and computes trading
performance metrics including Sharpe ratio, win rate, max drawdown,
and the composite score used for experiment keep/discard decisions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class BacktestResult:
    composite_score: float
    sharpe: float
    num_trades: int
    win_rate: float
    max_drawdown: float
    profit_factor: float
    total_return: float
    regime_sharpes: list[float]  # Sharpe per regime window
    meets_constraints: bool

    def summary(self) -> str:
        return (
            f"---\n"
            f"composite_score:  {self.composite_score:.6f}\n"
            f"sharpe:           {self.sharpe:.6f}\n"
            f"num_trades:       {self.num_trades}\n"
            f"win_rate:         {self.win_rate:.6f}\n"
            f"max_drawdown:     {self.max_drawdown:.6f}\n"
            f"profit_factor:    {self.profit_factor:.6f}\n"
            f"total_return:     {self.total_return:.6f}\n"
            f"regime_sharpes:   {[round(s, 4) for s in self.regime_sharpes]}\n"
        )


# ---------------------------------------------------------------------------
# Signal-to-trades conversion
# ---------------------------------------------------------------------------

def signals_to_trades(
    predictions: np.ndarray,
    forward_returns: np.ndarray,
    threshold: float = 0.55,
    commission_pct: float = 0.26,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert model predictions to trade P&Ls.

    Args:
        predictions: shape (N, 3) — softmax probabilities for [short, flat, long]
        forward_returns: shape (N,) — actual forward returns
        threshold: minimum probability to trigger a trade
        commission_pct: round-trip commission as percentage

    Returns:
        trade_returns: P&L per trade (after commission)
        trade_mask: boolean mask of which bars had trades
    """
    N = len(predictions)
    trade_returns = np.zeros(N, dtype=np.float64)
    trade_mask = np.zeros(N, dtype=bool)
    commission = commission_pct / 100.0

    for i in range(N):
        if np.isnan(forward_returns[i]):
            continue

        p_short, p_flat, p_long = predictions[i]

        if p_long > threshold and p_long > p_short:
            # Long trade
            trade_returns[i] = forward_returns[i] - commission
            trade_mask[i] = True
        elif p_short > threshold and p_short > p_long:
            # Short trade
            trade_returns[i] = -forward_returns[i] - commission
            trade_mask[i] = True

    return trade_returns, trade_mask


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_sharpe(returns: np.ndarray, annualize_factor: float = 365 * 24 * 4) -> float:
    """
    Annualized Sharpe ratio from per-trade returns.
    annualize_factor: bars per year at 15-min resolution (365 × 24 × 4).
    """
    if len(returns) < 2:
        return 0.0
    mu = np.mean(returns)
    sigma = np.std(returns, ddof=1)
    if sigma < 1e-10:
        return 0.0
    # Scale by sqrt(trades_per_year / trades_observed) for annualization
    return float(mu / sigma * math.sqrt(min(annualize_factor, len(returns))))


def compute_max_drawdown(returns: np.ndarray) -> float:
    """Max drawdown from cumulative return series."""
    equity = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / peak
    return float(np.max(drawdown)) if len(drawdown) > 0 else 0.0


def compute_profit_factor(returns: np.ndarray) -> float:
    """Gross profit / gross loss."""
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    if losses < 1e-10:
        return 10.0 if gains > 0 else 0.0
    return float(gains / losses)


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    val_dataset,
    val_regime_windows: list[tuple[int, int]],
    batch_size: int = 64,
    threshold: float = 0.55,
    commission_pct: float = 0.26,
    device: str = "cuda",
) -> BacktestResult:
    """
    Full evaluation pipeline: model → predictions → trades → metrics.

    Args:
        model: trained model with forward(x) → logits of shape (B, 3)
        val_dataset: SignalDataset for validation period
        val_regime_windows: list of (start, end) index ranges in val set
        batch_size: evaluation batch size
        threshold: probability threshold for triggering trades
        commission_pct: round-trip commission
        device: torch device

    Returns:
        BacktestResult with all metrics and constraint check
    """
    model.eval()
    from signalfusion.model.prepare import make_dataloader

    loader = make_dataloader(val_dataset, batch_size=batch_size, shuffle=False)

    all_predictions = []
    all_returns = []

    for x, y, r in loader:
        x = x.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        all_predictions.append(probs)
        all_returns.append(r.numpy())

    if not all_predictions:
        return BacktestResult(
            composite_score=0.0, sharpe=0.0, num_trades=0, win_rate=0.0,
            max_drawdown=1.0, profit_factor=0.0, total_return=0.0,
            regime_sharpes=[0.0] * len(val_regime_windows), meets_constraints=False,
        )

    predictions = np.concatenate(all_predictions, axis=0)
    forward_returns = np.concatenate(all_returns, axis=0)

    # Convert to trades
    trade_returns, trade_mask = signals_to_trades(
        predictions, forward_returns, threshold, commission_pct,
    )

    traded_returns = trade_returns[trade_mask]
    num_trades = int(trade_mask.sum())

    if num_trades < 5:
        return BacktestResult(
            composite_score=0.0, sharpe=0.0, num_trades=num_trades, win_rate=0.0,
            max_drawdown=1.0, profit_factor=0.0, total_return=0.0,
            regime_sharpes=[0.0] * len(val_regime_windows), meets_constraints=False,
        )

    # Overall metrics
    sharpe = compute_sharpe(traded_returns)
    max_dd = compute_max_drawdown(traded_returns)
    win_rate = float(np.mean(traded_returns > 0))
    profit_factor = compute_profit_factor(traded_returns)
    total_return = float(np.prod(1 + traded_returns) - 1)

    # Per-regime Sharpe (use trade index mapping)
    trade_indices = np.where(trade_mask)[0]
    regime_sharpes = []
    for start, end in val_regime_windows:
        regime_mask = (trade_indices >= start) & (trade_indices < end)
        regime_returns = traded_returns[regime_mask[: len(traded_returns)]]
        if len(regime_returns) >= 3:
            regime_sharpes.append(compute_sharpe(regime_returns))
        else:
            regime_sharpes.append(0.0)

    # Composite score
    if sharpe > 0 and num_trades >= 50 and max_dd < 0.15:
        composite = sharpe * math.sqrt(num_trades) * (1 - max_dd)
    else:
        composite = 0.0

    # Constraint check
    positive_regimes = sum(1 for s in regime_sharpes if s > 0)
    meets_constraints = (
        num_trades >= 50
        and max_dd < 0.15
        and sharpe > 0
        and positive_regimes >= 3
    )

    return BacktestResult(
        composite_score=composite,
        sharpe=sharpe,
        num_trades=num_trades,
        win_rate=win_rate,
        max_drawdown=max_dd,
        profit_factor=profit_factor,
        total_return=total_return,
        regime_sharpes=regime_sharpes,
        meets_constraints=meets_constraints,
    )
