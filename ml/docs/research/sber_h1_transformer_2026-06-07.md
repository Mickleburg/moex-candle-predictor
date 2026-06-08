# Transformer Architecture — SBER H1 Research Report

**Date**: 2026-06-07  
**Script**: `ml/scripts/sber_transformer_research.py`  
**Result file**: `ml/docs/research/sber_h1_transformer_results_20260607_210559.json`

## Hypothesis

Self-attention over 32-step candle sequences dynamically weights which past candles matter
for each prediction. Compared to LSTM's sequential (equal-weight) processing, Transformer
was expected to better capture long-range dependencies and improve WF macro-F1 by +0.02..0.05.

## Architecture

```
Input:   (batch, 32, 14)
Proj:    Linear(14 → 64)
PosEmb:  nn.Embedding(32, 64) — learned positional encoding
Encoder: TransformerEncoder(d_model=64, nhead=4, ffn=256, layers=2, dropout=0.1)
Pool:    mean over time dimension
Head:    Linear(64→32) → ReLU → Dropout(0.1) → Linear(32→3)
Params:  105,155
```

Training: Adam(lr=5e-4, wd=1e-4), CosineAnnealingLR, patience=10, max_epochs=60, batch=256.  
Same walk-forward protocol as LSTM v2 (4 folds, initial_train=12000, val=2000, seeds=[7,42,100]).

## Results

### Per-fold

| Fold | Seed | macro-F1 | SELL | HOLD | BUY | conf>0.50 | epochs |
|------|------|----------|------|------|-----|-----------|--------|
| 1 | 7   | 0.4750 | 0.408 | 0.604 | 0.412 | 21.6% | 19 |
| 1 | 42  | 0.4694 | 0.421 | 0.587 | 0.400 | 16.4% | 13 |
| 1 | 100 | 0.4759 | 0.363 | 0.607 | 0.458 | 25.2% | 21 |
| **1 mean** | | **0.4734** | | | | **21.1%** | |
| 2 | 7   | 0.4893 | 0.402 | 0.616 | 0.450 | 32.0% | 26 |
| 2 | 42  | 0.4751 | 0.395 | 0.590 | 0.441 | 23.6% | 32 |
| 2 | 100 | 0.4871 | 0.394 | 0.615 | 0.453 | 31.9% | 18 |
| **2 mean** | | **0.4838** | | | | **29.2%** | |
| 3 | 7   | 0.4770 | 0.483 | 0.601 | 0.347 | 30.2% | 18 |
| 3 | 42  | 0.4857 | 0.456 | 0.590 | 0.411 | 32.7% | 22 |
| 3 | 100 | 0.4905 | 0.476 | 0.594 | 0.402 | 39.4% | 37 |
| **3 mean** | | **0.4844** | | | | **34.1%** | |
| 4 | 7   | 0.4266 | 0.435 | 0.484 | 0.361 | 21.8% | 33 |
| 4 | 42  | 0.4356 | 0.449 | 0.493 | 0.365 | 19.8% | 24 |
| 4 | 100 | 0.4521 | 0.470 | 0.491 | 0.395 | 22.7% | 38 |
| **4 mean** | | **0.4381** | | | | **21.4%** | |

### Aggregate vs baselines

| Metric | Transformer | LSTM v2 | ET |
|--------|------------|---------|-----|
| WF macro-F1 | 0.4699 ± 0.0201 | **0.4778** ± 0.022 | 0.4738 |
| Worst fold | 0.4381 | 0.440 | — |
| conf>0.50 rate | **26.4%** | ~1% | — |
| Total time | 52 min | — | — |

## Interpretation

### 1. Transformer does not beat LSTM on F1

Delta = −0.008, within noise (σ=0.02). Transformer performs identically to ET baseline
and marginally below LSTM v2. The self-attention hypothesis is not confirmed for 1H OHLCV
on MOEX at this sequence length and architecture.

### 2. conf>0.50 = 26.4% is overconfidence, not signal

LSTM produces confident predictions in only 1% of cases, and those proved highly directional
(backtest Sharpe=6.38). Transformer produces 26× more "confident" predictions with *lower*
overall F1. This is a classic overconfidence failure:
- Mean pooling diffuses the probability mass across 32 time steps
- The model learns a soft-boundary representation → many outputs land slightly above 0.50
- These are not genuinely high-conviction predictions

Expected backtest outcome: too many trades (~2000 vs LSTM's 78), Sharpe likely negative.

### 3. Fold 4 collapse is structural, not architecture-specific

Fold 4 (most recent period, 2024–2025 MOEX) collapses to 0.4381 — same pattern as LSTM v2
(worst fold 0.440) and LSTM v1. This is a regime-change in recent MOEX data that no model
trained only on SBER 1H OHLCV can overcome.

### 4. Mean pooling is likely the wrong aggregation

For classification at the "end of sequence" (predict next candle), mean pooling treats all
32 historical steps equally. Better options:
- Last-step output: use `x[:, -1, :]` instead of `x.mean(dim=1)`
- CLS token: prepend a learnable class token
These may reduce overconfidence by focusing the prediction on the most recent context.

## Conclusion

**Transformer (mean pooling) does not improve over LSTM v2.**  
The architecture needs modification before it becomes useful. Two paths forward:

1. **Fix Transformer**: replace mean pooling with last-step or CLS-token aggregation
   → expected to reduce overconfidence and potentially improve F1
2. **Multi-ticker training**: train LSTM/Transformer on SBER+LKOH+GAZP jointly
   → 3× data, expected +0.01..0.03 WF F1 improvement

**Recommendation**: multi-ticker is higher-expected-value. The overconfidence problem
requires another experiment iteration with uncertain gain; multi-ticker addresses the
root cause (data volume ceiling).

## Updated Experiment Table

| Experiment | WF F1 | Δ vs LSTM | Conclusion |
|-----------|-------|-----------|------------|
| ET baseline | 0.4738 | −0.004 | Reference (frozen candidate) |
| Time ablation | 0.4097 | −0.068 | Keep time features |
| Calibration | 0.4503–0.4520 | −0.026 | Skip (ECE already OK) |
| SVD W2V | 0.4717 | −0.006 | Negative |
| Lag features + ET | 0.4711 | −0.007 | Negative |
| **LSTM v2** | **0.4778** | — | **Best model, artifact ready** |
| Target grid | 0.4738 | — | h=3:k=1.25 confirmed optimal |
| **Transformer (mean pool)** | **0.4699** | **−0.008** | **Overconfident, no gain** |
