# SignalFusion

Cross-modal attention transformer for crypto trading signals. Learns patterns
across 42 features from 6 data layers (market microstructure, derivatives,
on-chain, sentiment, macro, market structure) using channel-independent
encoders with cross-attention fusion.

Designed for the [autoresearch](https://github.com/karpathy/autoresearch) pattern:
an AI agent autonomously modifies the model architecture, trains for 5 minutes,
evaluates, keeps or discards, and repeats overnight.

## Architecture

```
Layer 1: Microstructure (6)  → Encoder → ┐
Layer 2: Derivatives (8)     → Encoder → ┤
Layer 3: On-Chain (9)        → Encoder → ┤→ Cross-Attention → Pool → Output Head → Signal
Layer 4: Sentiment (6)       → Encoder → ┤
Layer 5: Macro (8)           → Encoder → ┤
Layer 6: Market Structure (5)→ Encoder → ┘
```

Each layer gets an independent transformer encoder. Cross-attention fuses
representations across layers. The output head predicts trade direction
(short/flat/long) from the fused representation.

## Signal Inventory (42 features)

**Microstructure** (Kraken, free): OHLCV bars, trade count
**Derivatives** (Binance, free): Funding rate, open interest, long/short ratio, liquidations, taker buy ratio, basis, top trader positioning
**On-Chain** (CoinMetrics, free tier): Exchange flows, active addresses, MVRV, SOPR, whale transactions, miner outflows, stablecoin supply
**Sentiment** (CoinGecko/ApeWisdom, free): Fear & Greed, Reddit mentions/sentiment, Twitter velocity, Google Trends, news sentiment
**Macro** (FRED/Yahoo, free): DXY, 10Y yield, 2Y-10Y spread, VIX, gold, S&P 500, M2, Fed funds
**Market Structure** (CoinGecko, free): BTC dominance, ETH/BTC ratio, stablecoin market cap, exchange volume concentration, alt-BTC correlation

## Quick Start

```bash
# Install dependencies
pip install -e .

# Initialize database and backfill historical data
python -m signalfusion.collectors.run --backfill

# Start continuous data collection (run in background)
python -m signalfusion.collectors.run --daemon &

# Run a single training experiment
python -m signalfusion.model.architecture
```

## Autoresearch Mode

```bash
# Point your AI agent at program.md and let it go
# The agent modifies signalfusion/model/architecture.py
# Everything else is read-only
```

See `program.md` for full autonomous research instructions.

## Project Structure

```
signalfusion/
├── data/
│   └── schema.py              # Signal definitions (42 features, 6 layers)
├── collectors/
│   ├── base.py                # Collector interface + SQLite writer
│   ├── kraken_ohlcv.py        # Price bars (free, no API key)
│   ├── binance_derivatives.py # Derivatives data (free, no API key)
│   └── run.py                 # Collector orchestrator
├── model/
│   ├── prepare.py             # Data loading + normalization (DO NOT MODIFY)
│   └── architecture.py        # Model architecture (AGENT MODIFIES THIS)
└── evaluation/
    └── backtest.py            # Evaluation harness (DO NOT MODIFY)

program.md                     # Autoresearch instructions
```

## Hardware Requirements

Tested on Olares One (RTX 5090 24GB, 96GB RAM). The baseline model is ~2.5M
parameters and uses <1GB VRAM. Any NVIDIA GPU with 4GB+ VRAM works.

## License

MIT
