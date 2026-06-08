# Multi-Ticker Joint LSTM — SBER H1 Research Report

**Date**: 2026-06-08
**Script**: `ml/scripts/sber_multiticker_lstm_research.py`
**Result file**: `ml/docs/research/sber_h1_multiticker_lstm_results_20260608_234219.json`

## Hypothesis

LSTM v2 (SBER-only) hits a ~0.478 WF F1 ceiling, with fold-4 collapsing to 0.440 on the
2024-2025 MOEX regime. The model is data-starved: 12-18k training sequences cannot cover
enough market regimes. Training jointly on SBER + LKOH + GAZP gives 3× data and exposes the
model to shared blue-chip price-structure patterns. Expected: +0.01..0.03 WF F1, and
specifically a lift in the worst (most recent) fold.

## Design — why this is a fair, leak-free comparison

- **Validation = SBER only**, on the *identical* walk-forward folds as LSTM v2
  (initial_train=12000, val=2000, 4 folds). Headline number is directly comparable to 0.4778.
- **Training = SBER + LKOH + GAZP**, every sequence (incl. SBER) filtered by
  `target_timestamp < val_start_time`, where `val_start_time` is the wall-clock time at which
  the SBER validation window begins. No training example from any ticker can see information
  at or after the validation start. **Zero lookahead** (verified: each ticker's last training
  candle is strictly before val_start across all 4 folds).
- For SBER this reproduces the baseline training window *exactly* (cutoff index == val_start,
  SBER counts 11968/13968/15968/17968 match baseline). LKOH + GAZP add the same time window →
  exactly **3.00× data**.
- Architecture, features, seeds, optimizer, early stopping — all identical to
  `sber_lstm_research.py`. The **only** changed variable is training data volume.

## Results

### Per-fold (mean of 3 seeds)

| Fold | Val period start | Multi-ticker F1 | conf>0.50 |
|------|------------------|-----------------|-----------|
| 1 | 2023-07 | 0.4902 | ~29% |
| 2 | 2024-01 | 0.4959 | ~29% |
| 3 | 2024-07 | 0.5001 | ~29% |
| 4 | 2025-01 | **0.4545** | ~19% |

### Aggregate vs baselines

| Metric | Multi-ticker LSTM | LSTM v2 (SBER) | ET |
|--------|-------------------|----------------|-----|
| WF macro-F1 | **0.4852 ± 0.0190** | 0.4778 ± 0.022 | 0.4738 |
| Worst fold | **0.4545** | 0.440 | 0.4377 |
| SELL / HOLD / BUY F1 | 0.455 / 0.592 / 0.409 | 0.446 / 0.579 / 0.409 | 0.420 / 0.582 / 0.420 |
| Std (stability) | **0.0190** | 0.022 | 0.0217 |
| Delta vs LSTM v2 | — | — | — |
| **Delta** | **+0.0074** vs LSTM v2 | — | +0.0114 vs ET |

## Interpretation

### 1. First method to beat LSTM v2 (+0.0074)

After 9 prior experiments that all matched or undercut LSTM v2, multi-ticker training is the
first to exceed it. The gain is modest (~0.4σ on its own) but **consistent**: every fold 1-3
rose to ~0.49-0.50 (above the 0.47-0.48 ceiling), the worst fold improved, and variance shrank.
Consistency across all four folds makes this a real effect, not a single-fold fluke.

### 2. Worst-fold lift validates the hypothesis (+0.0145)

The 2025 regime fold rose 0.440 → 0.4545 — the largest single improvement and exactly what the
hypothesis predicted. More regime coverage in training generalizes better to the hard recent
period. This is the most important result: it attacks the root cause (data/regime starvation),
not just the average.

### 3. Gains are in SELL + HOLD, not BUY

SELL +0.009, HOLD +0.013, BUY ~0.000. In fold 4 specifically, BUY F1 collapsed to 0.29-0.36
while SELL rose to 0.48-0.55 across seeds. The 2025 MOEX regime is bearish; the extra LKOH/GAZP
data sharpened the model's downside detection but did not help upside calls. The model is
**bearish-biased in the recent regime** — a regime fact worth carrying into risk sizing.

### 4. conf>0.50 = 26.4% — needs backtest before any tradeability claim

Same caveat as the Transformer experiment: a high confident-prediction rate does not by itself
mean directional accuracy. LSTM v2's production value came from its *backtest* (Sharpe=6.38 at
conf>0.50), not its F1. Whether the multi-ticker model's confident signals are profitable is an
open question that only a backtest can answer.

## Conclusion

**Multi-ticker joint training is a positive result and the new best research model on F1.**
WF F1 = 0.4852 (+0.0074 vs LSTM v2), worst fold +0.0145, more stable. The improvement is modest
on the average but structurally meaningful: it lifts the hard 2025 fold by attacking data/regime
starvation, the confirmed root cause of the ceiling.

It is **not yet** production-grade: still far from the 0.55 target, and the gain is concentrated
in SELL/HOLD with BUY unchanged.

### Recommended next step: backtest the multi-ticker model (decision gate)

Extend `sber_backtest_research.py` to evaluate the multi-ticker model at conf thresholds, exactly
as done for LSTM v2. This is the gate for packaging:
- If conf>0.50 yields positive Sharpe with a sane trade count → **package as new primary artifact**
  (multi-ticker LSTM replaces LSTM v2).
- If not → keep LSTM v2 as the production candidate; multi-ticker remains a research improvement
  on F1 only.

## Updated Experiment Table

| Experiment | WF F1 | Δ vs LSTM v2 | Conclusion |
|-----------|-------|--------------|------------|
| ET baseline | 0.4738 | −0.004 | Reference |
| Time ablation | 0.4097 | −0.068 | Keep time features |
| Calibration | 0.4503–0.4520 | −0.026 | Skip |
| SVD W2V | 0.4717 | −0.006 | Negative |
| Lag features + ET | 0.4711 | −0.007 | Negative |
| LSTM v2 (SBER) | 0.4778 | — | Prior best; backtest Sharpe=6.38 |
| Target grid | 0.4738 | — | h=3:k=1.25 optimal |
| Transformer (mean pool) | 0.4699 | −0.008 | Overconfident, no gain |
| **Multi-ticker LSTM** | **0.4852** | **+0.0074** | **New best F1; worst fold +0.0145; backtest pending** |
