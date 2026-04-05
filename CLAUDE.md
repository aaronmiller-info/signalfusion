# SignalFusion — Full Project Context

## What This Is

SignalFusion is a **from-scratch cross-modal attention transformer** for crypto trading.
It trains on 42 features across 6 data layers (microstructure, derivatives, on-chain,
sentiment, macro, market structure) and predicts trade direction (short/flat/long).

The project follows Karpathy's **autoresearch pattern**: an AI agent modifies one file
(`architecture.py`), trains for 5 minutes on GPU, evaluates a single composite metric,
keeps or discards the change, and loops forever.

**Owner**: Aaron Miller (aaronmiller-info on GitHub).
**Hardware**: Olares One — RTX 5090 Mobile 24GB GDDR7, Intel Core Ultra 9 275HX (24-core),
96GB DDR5 RAM, 2TB NVMe. Ubuntu 24.04 with k3s.

## Current State (as of April 5, 2026)

### What's Running

- **Collector daemon** (`signalfusion-collector.service`): systemd service, auto-restarts,
  continuously fetching data from all sources. Check status:
  ```bash
  sudo systemctl status signalfusion-collector
  sudo journalctl -u signalfusion-collector -f
  ```

- **Database**: `data/signals.db` (SQLite with WAL mode)
  - 90 days of 15-min OHLCV backfilled (Jan 5 – Apr 5, 2026) via Binance.US
  - Live feeds accumulating from 5 collectors
  - ~56,000 rows across 6 symbols

- **Git branch**: `autoresearch/apr5` — branched from main, baseline logged in `results.tsv`

### Signal Coverage: 18/42 Active

| Layer | Signals | Active | Source | Notes |
|-------|---------|--------|--------|-------|
| Microstructure | 6 | 6 | Kraken (live), Binance.US (backfill) | Fully operational |
| Derivatives | 8 | 2 | OKX public API | funding_rate, open_interest. Other 6 stubbed None |
| On-Chain | 9 | 0 | — | Needs CoinMetrics or alternative. All None |
| Sentiment | 6 | 1 | alternative.me | fear_greed_index only. Others need auth APIs |
| Macro | 8 | 6 | Yahoo Finance | DXY, 10Y yield, spread, VIX, gold, S&P 500. M2 & fed funds need FRED key |
| Market Structure | 5 | 3 | CoinGecko free | BTC dominance, ETH/BTC, stablecoin mcap. HHI & correlation stubbed |

### Baseline Results

```
composite_score:  0.000000
sharpe:           0.000000
num_trades:       2
peak_vram_mb:     5267.1
training_seconds: 180.0
total_seconds:    356.5
num_params_M:     1.1
meets_constraints: False
```

The model predicts flat/neutral for almost everything. This is expected for a baseline
with only OHLCV + sparse non-price features. The autoresearch agent's job is to iterate
from here.

## Architecture Overview

### Files You Can Modify

- **`signalfusion/model/architecture.py`** — THE file the autoresearch agent modifies.
  Contains model architecture, hyperparameters, feature selection, training loop, loss
  function. Everything is fair game.

### Files That Are Read-Only (Do NOT Modify During Autoresearch)

- **`signalfusion/model/prepare.py`** — Data loading, normalization, train/val/holdout
  splits. Fixed evaluation harness.
  - Detects which features have actual data (active features)
  - Filters to only active columns, fills remaining NaN with 0
  - Returns `metadata["active_features"]` and `metadata["num_active_features"]`
  - Splits: 55% train / 20% val / 25% holdout (by time, not random)
  - Z-score lookback: 2880 bars (30 days at 15-min)

- **`signalfusion/evaluation/backtest.py`** — Evaluation harness. Converts predictions
  to trades, computes Sharpe, drawdown, profit factor, composite score.

- **`signalfusion/data/schema.py`** — Defines all 42 signals with Layer enum,
  NormMethod enum, SignalDef dataclass. Single source of truth for features.

- **Collector files** (`signalfusion/collectors/`) — Data pipeline is fixed.

### Key Design Decisions

1. **Partial feature support**: prepare.py auto-detects which of the 42 features have
   data and only passes those to the model. `get_active_features()` in architecture.py
   accepts the active feature name list from metadata and maps them to their layer
   groupings. This means the model architecture adapts to available data — as more
   collectors come online, more features activate automatically.

2. **Cross-modal attention**: Each signal layer (microstructure, derivatives, etc.) gets
   its own ChannelEncoder (transformer). Cross-attention blocks fuse representations
   across layers. When only one layer has data (e.g., just microstructure), there's
   no cross-attention — just the single encoder.

3. **Hybrid loss**: 70% classification cross-entropy + 30% differentiable Sharpe ratio.
   The classification branch learns direction, the Sharpe branch optimizes risk-adjusted
   returns directly.

4. **Composite score metric**:
   ```
   composite_score = sharpe × sqrt(num_trades) × (1 - max_drawdown)
   ```
   Constraints: ≥50 trades, <15% drawdown, positive Sharpe, positive Sharpe in ≥3 of
   4 regime windows (anti-overfit).

## Running Experiments

```bash
# Activate the venv
source /opt/signalfusion/.venv/bin/activate

# Run a training experiment (5-min time budget)
python -m signalfusion.model.architecture > run.log 2>&1

# Check results
grep "^composite_score:\|^sharpe:\|^num_trades:\|^peak_vram_mb:" run.log

# If it crashed, check the traceback
tail -50 run.log
```

### The Autoresearch Loop

See `program.md` for the full protocol. Summary:

1. Read architecture.py, understand current state
2. Make a change (architecture, features, loss, training, etc.)
3. `git commit`
4. Run: `python -m signalfusion.model.architecture > run.log 2>&1`
5. Parse results from run.log
6. If improved AND constraints met → keep (advance branch)
7. If worse → `git reset --hard HEAD~1` (revert)
8. Log to results.tsv
9. LOOP FOREVER — do not stop to ask the human

Expected pace: ~12 experiments/hour (~6 min per cycle: 5 min training + 1 min overhead).

### Results Logging

`results.tsv` (tab-separated, DO NOT commit):

```
commit	composite_score	sharpe	num_trades	peak_vram_gb	status	description
b946b98	0.000000	0.000000	2	5.1	keep	baseline: 18 features, default architecture
```

## Data Collection

### Collector Daemon (Already Running)

```bash
# Status
sudo systemctl status signalfusion-collector

# Logs
sudo journalctl -u signalfusion-collector -f

# Restart after code changes
cd /opt/signalfusion && git pull
sudo systemctl restart signalfusion-collector
```

### Manual Collection

```bash
# One-shot (all collectors)
python -m signalfusion.collectors.run

# Historical backfill (Binance.US OHLCV, 90 days)
python -m signalfusion.collectors.run --backfill
```

### Collector Sources

| Collector | File | API | Rate Limit | Frequency |
|-----------|------|-----|------------|-----------|
| Kraken OHLCV | `kraken_ohlcv.py` | Kraken public | Generous | 60s |
| OKX Derivatives | `binance_derivatives.py` | OKX public | Generous | 300s |
| Macro | `macro.py` | Yahoo Finance | No auth | 86400s |
| Sentiment | `sentiment.py` | alternative.me | No auth | 86400s |
| CoinGecko | `coingecko.py` | CoinGecko free | 30 req/min | 3600s |

**Important**: Binance.com and Bybit are **geo-blocked in the US**. We use:
- Binance.US (`api.binance.us`) for historical OHLCV backfill
- OKX (`www.okx.com`) for derivatives (funding rate, open interest)
- Kraken for live OHLCV

### Symbols

BTC/USD, ETH/USD, SOL/USD, XRP/USD, LINK/USD, AVAX/USD

All training currently targets BTC/USD (configurable via `SYMBOL` in architecture.py).

## Technical Notes

### Python Environment

```bash
# Venv location
/opt/signalfusion/.venv/bin/python

# Key packages
# PyTorch 2.11.0+cu130, numpy 2.4, pandas 3.0, httpx, scikit-learn
```

### GPU Details

- RTX 5090 Mobile, 24GB GDDR7 VRAM
- CUDA 13.0 via PyTorch 2.11
- Baseline uses ~5.3GB VRAM with 1.1M params and 18 active features
- Other services (STT, TTS, embeddings) may use ~8GB — stay under 14GB peak
- Check usage: `nvidia-smi`

### Database Schema

SQLite `data/signals.db`, single table:

```sql
signals(
  symbol TEXT NOT NULL,
  timestamp REAL NOT NULL,
  -- 42 feature columns (REAL, nullable):
  open, high, low, close, volume, trade_count,
  funding_rate, open_interest, open_interest_delta, long_short_ratio,
  liquidation_volume, top_trader_long_ratio, basis_annualized, taker_buy_ratio,
  exchange_inflow, exchange_outflow, exchange_net_flow, active_addresses,
  mvrv_ratio, sopr, whale_tx_count, miner_outflow, stablecoin_exchange_supply,
  fear_greed_index, reddit_mentions, reddit_sentiment,
  twitter_mention_velocity, google_trends, news_sentiment,
  dxy, us_10y_yield, us_2y10y_spread, vix, gold, sp500, global_m2, fed_funds_rate,
  btc_dominance, eth_btc_ratio, stablecoin_market_cap, exchange_volume_hhi,
  alt_btc_correlation,
  PRIMARY KEY (symbol, timestamp)
)
```

Collectors write at their native timestamps. The prepare pipeline handles alignment
and resampling to 15-minute bars.

### Git Configuration

```
user.email = penny@aaronmiller.info
user.name = Penny
remote = https://github.com/aaronmiller-info/signalfusion.git
```

## Research Directions (Priority Order)

1. **Feature selection**: With only 18/42 signals active, some may be noise. Ablate
   aggressively — which features actually help the model trade better?

2. **Model sizing**: Current 1.1M params may be too large for 6 real features.
   Try smaller models (D_MODEL=32, fewer layers) or larger if data supports it.

3. **Training dynamics**: The baseline barely trades (2 trades). The model is likely
   too conservative. Try:
   - Lower classification threshold (currently 0.55 in backtest.py... but that's
     read-only, so adjust the model's confidence distribution instead)
   - More aggressive loss weighting toward Sharpe
   - Different learning rates, warmup schedules

4. **Prediction horizon**: Currently 4 bars (1 hour). Try 1 bar (15 min), 8 bars (2h),
   16 bars (4h), 96 bars (24h).

5. **Architecture**: Try simpler architectures first — a single encoder without
   cross-attention may outperform when most layers have no data.

6. **Sequence length**: Currently 200 bars (50 hours). May be too long for 15-min
   resolution. Try 50, 100, 400.

7. **Regularization**: Dropout (0.1 currently), weight decay (1e-4), gradient clipping.

8. **Loss function**: Pure classification, pure Sharpe, different hybrid ratios.

## Known Issues

- **Sparse non-OHLCV data**: Most non-price features only have data from April 5 onward
  (when collectors were first run). The 90-day backfill is OHLCV only. As the daemon
  runs, non-price features will accumulate. The model handles this via NaN→0 filling.

- **Yahoo Finance quirks**: The chart API sometimes returns None for the current day's
  close if markets haven't closed yet. The macro collector walks backwards to find the
  last valid close.

- **CoinGecko rate limits**: Free tier is 30 req/min. The collector adds 2.5s delays
  between calls. May hit limits if other services also use CoinGecko.

- **No on-chain data yet**: The 9 on-chain signals (exchange flows, MVRV, SOPR, etc.)
  have no collector. CoinMetrics community API or Glassnode would fill this gap.

## Useful Commands

```bash
# Check DB size and row counts
python3 -c "
import sqlite3
conn = sqlite3.connect('data/signals.db')
for s, c in conn.execute('SELECT symbol, COUNT(*) FROM signals GROUP BY symbol').fetchall():
    print(f'{s}: {c} rows')
conn.close()
"

# Check which features have data
python3 -c "
import sqlite3
conn = sqlite3.connect('data/signals.db')
cols = [r[1] for r in conn.execute('PRAGMA table_info(signals)').fetchall()]
for col in cols[2:]:
    count = conn.execute(f'SELECT COUNT(*) FROM signals WHERE \"{col}\" IS NOT NULL').fetchone()[0]
    if count > 0: print(f'  {col}: {count} rows')
conn.close()
"

# Quick GPU check
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader

# Check daemon is collecting
sudo journalctl -u signalfusion-collector --since '5 min ago' --no-pager
```
