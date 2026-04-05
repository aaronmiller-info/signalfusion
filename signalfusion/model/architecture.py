"""
SignalFusion model architecture — THE FILE THE AGENT MODIFIES.

Cross-modal attention transformer for crypto trading signals.
Everything is fair game: architecture, features, training, loss function.

Usage: python -m signalfusion.model.architecture
"""

import gc
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from signalfusion.data.schema import NUM_FEATURES, SIGNALS_BY_LAYER, Layer, FEATURE_NAMES
from signalfusion.model.prepare import (
    TIME_BUDGET, prepare_datasets, make_dataloader,
)
from signalfusion.evaluation.backtest import evaluate_model

# ---------------------------------------------------------------------------
# Hyperparameters (edit freely)
# ---------------------------------------------------------------------------

# Architecture
D_MODEL = 64                    # embedding dimension
N_HEADS = 4                     # attention heads per layer
N_ENCODER_LAYERS = 2            # layers per channel encoder
N_CROSS_LAYERS = 2              # cross-attention fusion layers
DROPOUT = 0.1                   # dropout rate
SEQ_LEN = 200                   # lookback window in bars

# Training
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
WARMUP_STEPS = 50
MAX_EPOCHS = 100                # will be cut short by time budget

# Loss (hybrid: classification + Sharpe)
CLASSIFICATION_WEIGHT = 0.7     # weight for cross-entropy loss
SHARPE_WEIGHT = 0.3             # weight for differentiable Sharpe loss

# Prediction
NUM_CLASSES = 3                 # short, flat, long
FORWARD_BARS = 4                # 1 hour at 15-min bars
THRESHOLD_PCT = 0.5             # ±0.5% for labels

# Feature selection — which layers to include
ACTIVE_LAYERS = [
    Layer.MICROSTRUCTURE,
    Layer.DERIVATIVES,
    Layer.ON_CHAIN,
    Layer.SENTIMENT,
    Layer.MACRO,
    Layer.MARKET_STRUCTURE,
]

# Which symbol to train/eval on
SYMBOL = "BTC/USD"


# ---------------------------------------------------------------------------
# Feature grouping
# ---------------------------------------------------------------------------

def get_active_features() -> dict[Layer, list[int]]:
    """Map each active layer to its feature indices in the 42-feature vector."""
    layer_indices = {}
    for layer in ACTIVE_LAYERS:
        indices = []
        for sig in SIGNALS_BY_LAYER[layer]:
            idx = FEATURE_NAMES.index(sig.name)
            indices.append(idx)
        layer_indices[layer] = indices
    return layer_indices


# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------

class ChannelEncoder(nn.Module):
    """Per-channel transformer encoder for one signal layer."""

    def __init__(self, n_features: int, d_model: int, n_heads: int,
                 n_layers: int, dropout: float, seq_len: int):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, n_features) → (B, T, d_model)"""
        x = self.input_proj(x) + self.pos_encoding[:, :x.size(1)]
        x = self.encoder(x)
        return self.norm(x)


class CrossAttentionBlock(nn.Module):
    """Bidirectional cross-attention between two sequences."""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """query attends to context. Returns updated query."""
        attended, _ = self.cross_attn(query, context, context)
        query = self.norm1(query + attended)
        query = self.norm2(query + self.ffn(query))
        return query


class SignalFusionModel(nn.Module):
    """
    Cross-modal attention transformer.

    Each signal layer gets an independent encoder. Cross-attention fuses
    representations across layers. Output head predicts trade direction.
    """

    def __init__(
        self,
        layer_features: dict[Layer, list[int]],
        d_model: int = D_MODEL,
        n_heads: int = N_HEADS,
        n_encoder_layers: int = N_ENCODER_LAYERS,
        n_cross_layers: int = N_CROSS_LAYERS,
        dropout: float = DROPOUT,
        seq_len: int = SEQ_LEN,
        num_classes: int = NUM_CLASSES,
    ):
        super().__init__()
        self.layer_features = layer_features
        self.layer_names = list(layer_features.keys())
        self.d_model = d_model

        # Per-channel encoders
        self.encoders = nn.ModuleDict()
        for layer in self.layer_names:
            n_feat = len(layer_features[layer])
            self.encoders[layer.value] = ChannelEncoder(
                n_feat, d_model, n_heads, n_encoder_layers, dropout, seq_len,
            )

        # Cross-attention layers — each channel attends to concatenation of all others
        self.cross_layers = nn.ModuleList()
        for _ in range(n_cross_layers):
            layer_blocks = nn.ModuleDict()
            for layer in self.layer_names:
                layer_blocks[layer.value] = CrossAttentionBlock(d_model, n_heads, dropout)
            self.cross_layers.append(layer_blocks)

        # Output: pool across channels and time, then classify
        n_channels = len(self.layer_names)
        self.output_norm = nn.LayerNorm(d_model * n_channels)
        self.output_head = nn.Sequential(
            nn.Linear(d_model * n_channels, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, NUM_FEATURES) — full 42-feature input
        Returns: (B, num_classes) — logits
        """
        # Encode each channel independently
        channel_reps = {}
        for layer in self.layer_names:
            feat_indices = self.layer_features[layer]
            channel_input = x[:, :, feat_indices]  # (B, T, n_feat_for_layer)
            channel_reps[layer] = self.encoders[layer.value](channel_input)

        # Cross-attention: each channel attends to all others
        for cross_block_dict in self.cross_layers:
            new_reps = {}
            for layer in self.layer_names:
                # Build context: concatenate all OTHER channels along time dim
                others = [channel_reps[l] for l in self.layer_names if l != layer]
                if others:
                    context = torch.cat(others, dim=1)  # (B, T*n_other, d_model)
                else:
                    context = channel_reps[layer]
                new_reps[layer] = cross_block_dict[layer.value](
                    channel_reps[layer], context,
                )
            channel_reps = new_reps

        # Pool: take last timestep from each channel, concatenate
        pooled = []
        for layer in self.layer_names:
            pooled.append(channel_reps[layer][:, -1, :])  # (B, d_model)
        pooled = torch.cat(pooled, dim=-1)  # (B, d_model * n_channels)

        pooled = self.output_norm(pooled)
        logits = self.output_head(pooled)
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def classification_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Standard cross-entropy for direction prediction."""
    return F.cross_entropy(logits, labels)


def differentiable_sharpe_loss(
    logits: torch.Tensor,
    forward_returns: torch.Tensor,
) -> torch.Tensor:
    """
    Differentiable Sharpe ratio loss.

    Convert logits to position signal (-1 to +1), multiply by returns,
    then compute negative Sharpe as loss.
    """
    probs = F.softmax(logits, dim=-1)
    # Position: P(long) - P(short) → signal in [-1, 1]
    position = probs[:, 2] - probs[:, 0]

    # Mask out NaN returns
    valid = ~torch.isnan(forward_returns)
    if valid.sum() < 10:
        return torch.tensor(0.0, device=logits.device)

    position = position[valid]
    returns = forward_returns[valid]
    pnl = position * returns

    mu = pnl.mean()
    sigma = pnl.std() + 1e-8
    sharpe = mu / sigma
    return -sharpe  # minimize negative Sharpe = maximize Sharpe


def hybrid_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    forward_returns: torch.Tensor,
    cls_weight: float = CLASSIFICATION_WEIGHT,
    sharpe_weight: float = SHARPE_WEIGHT,
) -> torch.Tensor:
    """Weighted combination of classification and Sharpe losses."""
    cls = classification_loss(logits, labels)
    sharpe = differentiable_sharpe_loss(logits, forward_returns)
    return cls_weight * cls + sharpe_weight * sharpe


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train():
    t_start = time.time()

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Data
    print("Loading data...")
    train_ds, val_ds, holdout_ds, metadata = prepare_datasets(
        symbol=SYMBOL, seq_len=SEQ_LEN, forward_bars=FORWARD_BARS,
        threshold_pct=THRESHOLD_PCT,
    )
    print(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")

    if len(train_ds) == 0 or len(val_ds) == 0:
        print("FAIL: insufficient data")
        exit(1)

    train_loader = make_dataloader(train_ds, BATCH_SIZE, shuffle=True)

    # Model
    layer_features = get_active_features()
    model = SignalFusionModel(
        layer_features=layer_features,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_encoder_layers=N_ENCODER_LAYERS,
        n_cross_layers=N_CROSS_LAYERS,
        dropout=DROPOUT,
        seq_len=SEQ_LEN,
        num_classes=NUM_CLASSES,
    ).to(device)

    num_params = model.count_parameters()
    print(f"Parameters: {num_params:,} ({num_params / 1e6:.1f}M)")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
    )

    # LR schedule with warmup
    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return step / max(WARMUP_STEPS, 1)
        progress = (step - WARMUP_STEPS) / max(1, MAX_EPOCHS * len(train_loader) - WARMUP_STEPS)
        return max(0.01, 0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training
    print(f"Time budget: {TIME_BUDGET}s")
    t_train_start = time.time()
    total_training_time = 0.0
    step = 0
    best_val_loss = float("inf")

    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    for epoch in range(MAX_EPOCHS):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0

        for x, labels, returns in train_loader:
            t0 = time.time()

            x = x.to(device)
            labels = labels.to(device)
            returns = returns.to(device)

            # Replace NaN with 0 in input features
            x = torch.nan_to_num(x, nan=0.0)

            logits = model(x)
            loss = hybrid_loss(logits, labels, returns)

            if torch.isnan(loss) or loss.item() > 100:
                print("FAIL: loss exploded")
                exit(1)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            t1 = time.time()
            total_training_time += (t1 - t0)
            epoch_loss += loss.item()
            epoch_steps += 1
            step += 1

            # Time check
            if total_training_time >= TIME_BUDGET * 0.6:
                break

        avg_loss = epoch_loss / max(epoch_steps, 1)
        elapsed = time.time() - t_train_start
        print(f"Epoch {epoch:3d} | loss: {avg_loss:.6f} | "
              f"lr: {scheduler.get_last_lr()[0]:.6f} | "
              f"elapsed: {elapsed:.0f}s | budget remaining: {TIME_BUDGET - total_training_time:.0f}s")

        if total_training_time >= TIME_BUDGET * 0.6:
            print(f"Training time budget reached ({total_training_time:.0f}s), moving to eval.")
            break

    # Evaluation
    print("\nEvaluating on validation set...")
    result = evaluate_model(
        model, val_ds, metadata["val_regime_windows"],
        batch_size=BATCH_SIZE, device=device,
    )

    # Summary
    t_end = time.time()
    peak_vram = torch.cuda.max_memory_allocated() / 1024 / 1024 if device == "cuda" else 0
    print(result.summary())
    print(f"peak_vram_mb:     {peak_vram:.1f}")
    print(f"training_seconds: {total_training_time:.1f}")
    print(f"total_seconds:    {t_end - t_start:.1f}")
    print(f"num_params_M:     {num_params / 1e6:.1f}")
    print(f"meets_constraints: {result.meets_constraints}")


if __name__ == "__main__":
    train()
