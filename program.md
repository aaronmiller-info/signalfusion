# SignalFusion — Autoresearch Program

This is an autonomous research loop for a cross-modal attention transformer
that learns crypto trading signals from 42 features across 6 data layers.

## Setup

To set up a new experiment run, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr5`). The branch
   `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main.
3. **Read the in-scope files**:
   - `README.md` — project context and signal inventory.
   - `signalfusion/model/architecture.py` — the file you modify. Model architecture,
     feature selection, training procedure.
   - `signalfusion/model/prepare.py` — fixed constants, data loading, normalization,
     train/val/holdout splits. Do not modify.
   - `signalfusion/evaluation/backtest.py` — fixed evaluation harness. Do not modify.
4. **Verify data exists**: Check that `data/signals.db` contains recent data. If not,
   tell the human to run `python -m signalfusion.collectors.run`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The
   baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

## Experimentation

Each experiment runs on a single GPU (RTX 5090, 24GB VRAM). Training and evaluation
run for a **fixed time budget of 5 minutes** total. Launch via:

```bash
python -m signalfusion.model.architecture
```

### What you CAN do

- Modify `signalfusion/model/architecture.py` — this is the only file you edit.
  Everything is fair game:
  - Model architecture (layers, heads, d_model, attention patterns)
  - Which of the 42 input features to include or exclude
  - Feature embedding strategy (linear, conv1d, patch-based)
  - Cross-attention configuration (which layers cross-attend, how many heads)
  - Training procedure (learning rate, optimizer, batch size, epochs, warmup)
  - Loss function (classification, differentiable Sharpe, hybrid, custom)
  - Normalization strategy (z-score window, log returns, rank transform)
  - Regularization (dropout, weight decay, early stopping criteria)
  - Prediction horizon (1h, 4h, 24h)
  - Lookback window length
  - Output head design

### What you CANNOT do

- Modify `signalfusion/model/prepare.py`. It is read-only. Contains fixed data loading,
  normalization, and train/val/holdout split logic.
- Modify `signalfusion/evaluation/backtest.py`. It is read-only. Contains the fixed
  evaluation harness and metric computation.
- Modify any collector code. Data pipeline is fixed.
- Install new packages or add dependencies.
- Access the holdout set directly. Only `prepare.py` touches it, and only during
  final evaluation (which you don't control).

### The goal

**Maximize the composite score on the validation set:**

```
composite_score = sharpe × sqrt(num_trades) × (1 - max_drawdown)
```

Subject to constraints:
- `num_trades >= 50` (statistical significance)
- `max_drawdown < 0.15` (risk limit)
- `sharpe > 0.0` on validation (not just training)
- Positive Sharpe in at least 3 of 4 regime windows (anti-overfit)

**VRAM** is a soft constraint. The Olares runs other services (~8GB used). Stay
under 14GB peak to avoid OOM. The model should be small — 1-10M parameters.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement
that adds ugly complexity is not worth it. Removing something and getting equal or
better results is a great outcome. A 0.01 composite_score improvement from 30 lines
of hacky code? Probably not worth it. The same improvement from deleting code? Keep.

### The first run

Your very first run should always be to establish the baseline by running the
architecture as-is. Do not modify anything for the first run.

## Output format

The training script prints a summary:

```
---
composite_score:  1.234567
sharpe:           0.890000
num_trades:       127
win_rate:         0.534000
max_drawdown:     0.078000
profit_factor:    1.340000
regime_sharpes:   [0.45, 0.67, 1.12, 0.23]
peak_vram_mb:     2048.0
training_seconds: 180.2
total_seconds:    295.1
num_params_M:     2.5
```

Extract metrics: `grep "^composite_score:\|^sharpe:\|^peak_vram_mb:" run.log`

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated).

Header and 6 columns:

```
commit	composite_score	sharpe	num_trades	peak_vram_gb	status	description
```

1. git commit hash (short, 7 chars)
2. composite_score (e.g. 1.234567) — use 0.000000 for crashes
3. sharpe ratio (e.g. 0.890000)
4. num_trades (integer)
5. peak VRAM in GB (divide peak_vram_mb by 1024, round to .1f)
6. status: `keep`, `discard`, or `crash`
7. short text description of what this experiment tried

## The experiment loop

LOOP FOREVER:

1. Look at the git state: current branch/commit
2. Modify `architecture.py` with an experimental idea
3. git commit
4. Run: `python -m signalfusion.model.architecture > run.log 2>&1`
5. Read results: `grep "^composite_score:\|^sharpe:\|^num_trades:\|^peak_vram_mb:" run.log`
6. If grep is empty → crash. Run `tail -n 50 run.log` for traceback. Fix if trivial,
   else give up on that idea.
7. Record in results.tsv (do NOT commit results.tsv)
8. If composite_score improved AND constraints met → keep (advance branch)
9. If worse or constraints violated → `git reset --hard HEAD~1` (revert)

**Timeout**: If a run exceeds 10 minutes, kill it and treat as failure.

**Crashes**: Typos/imports → fix and re-run. Fundamentally broken idea → skip, log
"crash", move on.

**NEVER STOP**: Once the loop begins, do NOT pause to ask the human. The human may be
asleep. Continue working indefinitely until manually interrupted. If you run out of
ideas, re-read the signal inventory in README.md, try combining previous near-misses,
try more radical architectural changes, or ablate features to find which matter most.

Expected pace: ~12 experiments/hour, ~100 overnight.

## Research directions to explore

Rough priority order, but use your judgment:

1. **Feature selection**: Which of the 42 signals actually help? Ablate aggressively.
2. **Cross-attention topology**: Which channel pairs benefit from cross-attention?
   Maybe derivatives→price matters but sentiment→on-chain doesn't.
3. **Patch size**: PatchTST-style patching vs individual timestep tokens.
4. **Loss function**: Classification vs differentiable Sharpe vs hybrid weighting.
5. **Prediction horizon**: 1h vs 4h vs 24h — which has the best signal-to-noise?
6. **Model scaling**: Does going from 2M to 10M params help, or does it overfit?
7. **Temporal encoding**: Learned vs sinusoidal positional encoding. Time-of-day,
   day-of-week embeddings.
8. **Regularization**: Dropout rates, weight decay, gradient clipping, early stopping.
9. **Normalization**: z-score vs rank transform vs log returns for different channels.
10. **Ensemble**: If you find multiple good architectures, try a simple average.
