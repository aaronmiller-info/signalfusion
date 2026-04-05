# SignalFusion Development Guide

## What this is

A cross-modal attention transformer for crypto trading. The autoresearch pattern:
an AI agent modifies `signalfusion/model/architecture.py`, trains for 5 minutes,
evaluates, keeps or discards, repeats.

## Key rules

1. **Only modify `signalfusion/model/architecture.py`** during autoresearch.
2. `prepare.py` and `backtest.py` are READ-ONLY. They are the fixed evaluation harness.
3. `data/schema.py` defines all 42 signals. Read it for feature reference.
4. Metric: `composite_score = sharpe × sqrt(num_trades) × (1 - max_drawdown)`
5. Constraints: ≥50 trades, <15% drawdown, positive Sharpe in ≥3 of 4 regime windows.
6. VRAM budget: stay under 14GB peak (other services use the GPU).
7. See `program.md` for the full autoresearch loop instructions.

## Running experiments

```bash
python -m signalfusion.model.architecture > run.log 2>&1
grep "^composite_score:\|^sharpe:\|^peak_vram_mb:" run.log
```

## Data collection

```bash
python -m signalfusion.collectors.run --backfill    # historical
python -m signalfusion.collectors.run --daemon       # continuous
```
