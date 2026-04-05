"""
Signal schema — defines all 42 features across 6 layers.

Each signal has a name, layer, native frequency, normalization method,
and the collector responsible for fetching it. This is the single source
of truth for what the model can see.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Layer(str, Enum):
    MICROSTRUCTURE = "microstructure"
    DERIVATIVES = "derivatives"
    ON_CHAIN = "on_chain"
    SENTIMENT = "sentiment"
    MACRO = "macro"
    MARKET_STRUCTURE = "market_structure"


class NormMethod(str, Enum):
    ZSCORE = "zscore"           # rolling z-score (default for most)
    LOG_RETURN = "log_return"   # log(x_t / x_{t-1}) — for prices
    RANK = "rank"               # percentile rank in rolling window — for bounded signals
    RAW = "raw"                 # no normalization — for ratios already bounded [0, 1]
    DIFF = "diff"               # first difference — for cumulative metrics


@dataclass(frozen=True)
class SignalDef:
    name: str
    layer: Layer
    frequency_minutes: int      # native update cadence
    norm: NormMethod
    collector: str              # which collector module fetches this
    description: str


# ---------------------------------------------------------------------------
# The 42 signals
# ---------------------------------------------------------------------------

SIGNALS: list[SignalDef] = [
    # === Layer 1: Market Microstructure (6 signals) ===
    SignalDef("open", Layer.MICROSTRUCTURE, 1, NormMethod.LOG_RETURN, "kraken_ohlcv",
             "Bar open price"),
    SignalDef("high", Layer.MICROSTRUCTURE, 1, NormMethod.LOG_RETURN, "kraken_ohlcv",
             "Bar high price"),
    SignalDef("low", Layer.MICROSTRUCTURE, 1, NormMethod.LOG_RETURN, "kraken_ohlcv",
             "Bar low price"),
    SignalDef("close", Layer.MICROSTRUCTURE, 1, NormMethod.LOG_RETURN, "kraken_ohlcv",
             "Bar close price"),
    SignalDef("volume", Layer.MICROSTRUCTURE, 1, NormMethod.ZSCORE, "kraken_ohlcv",
             "Bar trading volume"),
    SignalDef("trade_count", Layer.MICROSTRUCTURE, 1, NormMethod.ZSCORE, "kraken_ohlcv",
             "Number of trades in bar"),

    # === Layer 2: Derivatives (8 signals) ===
    SignalDef("funding_rate", Layer.DERIVATIVES, 480, NormMethod.ZSCORE, "binance_derivatives",
             "Perpetual swap funding rate"),
    SignalDef("open_interest", Layer.DERIVATIVES, 60, NormMethod.ZSCORE, "binance_derivatives",
             "Total open interest in USD"),
    SignalDef("open_interest_delta", Layer.DERIVATIVES, 60, NormMethod.DIFF, "binance_derivatives",
             "Change in open interest"),
    SignalDef("long_short_ratio", Layer.DERIVATIVES, 5, NormMethod.ZSCORE, "binance_derivatives",
             "Long/short account ratio"),
    SignalDef("liquidation_volume", Layer.DERIVATIVES, 60, NormMethod.ZSCORE,
             "binance_derivatives", "Forced liquidation volume in USD"),
    SignalDef("top_trader_long_ratio", Layer.DERIVATIVES, 5, NormMethod.RAW,
             "binance_derivatives", "Top trader long position ratio"),
    SignalDef("basis_annualized", Layer.DERIVATIVES, 60, NormMethod.ZSCORE,
             "binance_derivatives", "Annualized futures basis (contango/backwardation)"),
    SignalDef("taker_buy_ratio", Layer.DERIVATIVES, 5, NormMethod.ZSCORE, "binance_derivatives",
             "Taker buy volume / total volume"),

    # === Layer 3: On-Chain (9 signals) ===
    SignalDef("exchange_inflow", Layer.ON_CHAIN, 60, NormMethod.ZSCORE, "coinmetrics",
             "Coins flowing to exchanges"),
    SignalDef("exchange_outflow", Layer.ON_CHAIN, 60, NormMethod.ZSCORE, "coinmetrics",
             "Coins flowing from exchanges"),
    SignalDef("exchange_net_flow", Layer.ON_CHAIN, 60, NormMethod.ZSCORE, "coinmetrics",
             "Net exchange flow (inflow - outflow)"),
    SignalDef("active_addresses", Layer.ON_CHAIN, 1440, NormMethod.ZSCORE, "coinmetrics",
             "Daily active addresses"),
    SignalDef("mvrv_ratio", Layer.ON_CHAIN, 1440, NormMethod.ZSCORE, "coinmetrics",
             "Market value to realized value ratio"),
    SignalDef("sopr", Layer.ON_CHAIN, 1440, NormMethod.ZSCORE, "coinmetrics",
             "Spent output profit ratio"),
    SignalDef("whale_tx_count", Layer.ON_CHAIN, 1440, NormMethod.ZSCORE, "coinmetrics",
             "Transactions > $1M"),
    SignalDef("miner_outflow", Layer.ON_CHAIN, 1440, NormMethod.ZSCORE, "coinmetrics",
             "Miner-attributed outflows"),
    SignalDef("stablecoin_exchange_supply", Layer.ON_CHAIN, 1440, NormMethod.ZSCORE,
             "coinmetrics", "Stablecoin balance on exchanges (dry powder)"),

    # === Layer 4: Sentiment (6 signals) ===
    SignalDef("fear_greed_index", Layer.SENTIMENT, 1440, NormMethod.RAW, "sentiment",
             "Crypto Fear & Greed Index [0-100]"),
    SignalDef("reddit_mentions", Layer.SENTIMENT, 60, NormMethod.ZSCORE, "sentiment",
             "Reddit mention count for symbol"),
    SignalDef("reddit_sentiment", Layer.SENTIMENT, 60, NormMethod.RAW, "sentiment",
             "Reddit sentiment score [-1, 1]"),
    SignalDef("twitter_mention_velocity", Layer.SENTIMENT, 60, NormMethod.ZSCORE, "sentiment",
             "Rate of change in Twitter/X mentions"),
    SignalDef("google_trends", Layer.SENTIMENT, 1440, NormMethod.ZSCORE, "sentiment",
             "Google search interest [0-100]"),
    SignalDef("news_sentiment", Layer.SENTIMENT, 60, NormMethod.RAW, "sentiment",
             "NLP-scored headline sentiment [-1, 1]"),

    # === Layer 5: Macro (8 signals) ===
    SignalDef("dxy", Layer.MACRO, 1440, NormMethod.LOG_RETURN, "macro",
             "US Dollar Index"),
    SignalDef("us_10y_yield", Layer.MACRO, 1440, NormMethod.DIFF, "macro",
             "US 10-year Treasury yield"),
    SignalDef("us_2y10y_spread", Layer.MACRO, 1440, NormMethod.DIFF, "macro",
             "2Y-10Y yield spread"),
    SignalDef("vix", Layer.MACRO, 1440, NormMethod.ZSCORE, "macro",
             "CBOE Volatility Index"),
    SignalDef("gold", Layer.MACRO, 1440, NormMethod.LOG_RETURN, "macro",
             "Gold spot price (XAU/USD)"),
    SignalDef("sp500", Layer.MACRO, 1440, NormMethod.LOG_RETURN, "macro",
             "S&P 500 index"),
    SignalDef("global_m2", Layer.MACRO, 43200, NormMethod.LOG_RETURN, "macro",
             "Global M2 money supply"),
    SignalDef("fed_funds_rate", Layer.MACRO, 43200, NormMethod.DIFF, "macro",
             "Federal funds effective rate"),

    # === Layer 6: Market Structure (5 signals) ===
    SignalDef("btc_dominance", Layer.MARKET_STRUCTURE, 60, NormMethod.DIFF, "coingecko",
             "BTC market cap as % of total crypto"),
    SignalDef("eth_btc_ratio", Layer.MARKET_STRUCTURE, 60, NormMethod.LOG_RETURN, "coingecko",
             "ETH/BTC price ratio"),
    SignalDef("stablecoin_market_cap", Layer.MARKET_STRUCTURE, 1440, NormMethod.LOG_RETURN,
             "coingecko", "Total stablecoin market cap"),
    SignalDef("exchange_volume_hhi", Layer.MARKET_STRUCTURE, 1440, NormMethod.ZSCORE,
             "coingecko", "Exchange volume concentration (HHI)"),
    SignalDef("alt_btc_correlation", Layer.MARKET_STRUCTURE, 1440, NormMethod.RAW, "coingecko",
             "Rolling 30d correlation of top alts with BTC"),
]

# Indexes for fast lookup
SIGNAL_BY_NAME: dict[str, SignalDef] = {s.name: s for s in SIGNALS}
SIGNALS_BY_LAYER: dict[Layer, list[SignalDef]] = {}
for _s in SIGNALS:
    SIGNALS_BY_LAYER.setdefault(_s.layer, []).append(_s)

FEATURE_NAMES: list[str] = [s.name for s in SIGNALS]
NUM_FEATURES: int = len(SIGNALS)  # 42

# Symbols we trade and collect data for
SYMBOLS: list[str] = ["BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD", "LINK/USD", "AVAX/USD"]
NUM_SYMBOLS: int = len(SYMBOLS)

# Common timeframe for all signals after resampling
RESAMPLE_MINUTES: int = 15
