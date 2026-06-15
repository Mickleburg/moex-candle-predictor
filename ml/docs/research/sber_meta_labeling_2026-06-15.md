# SBER Meta-Labeling — Stage 3 (#2) — 2026-06-15

**Script**: `ml/scripts/sber_meta_labeling.py`
**Result**: `ml/docs/research/sber_meta_labeling_20260615_015144.json`
**Input**: cached SBER LSTM predictions (no retrain). Light (seconds).

## Method

Lopez de Prado meta-labeling: primary LSTM gives direction; a secondary HistGBM decides
bet/no-bet on each BUY-argmax candidate. Past-only meta-features (primary probs/confidence,
volatility, session hour/dow/is_friday/is_evening, OHLCV momentum); meta-label = "would the
3h+stop long have won?". Time-ordered 50/50 split: train meta on the first half, evaluate on the
held-out second half vs the conf>0.50 baseline (both skip weekends).

## Result — NEGATIVE. Meta does not beat raw confidence.

- **Meta-model test AUC = 0.569** — barely above chance. The features hardly predict which BUY
  signal wins.

| Rule (held-out half, 1145 candidates) | Return | Sharpe | Win | Trades |
|---------------------------------------|--------|--------|-----|--------|
| **baseline conf>0.50** | **+27.14%** | **13.39** | 65.5% | 55 |
| meta P(win)>0.50 | +7.94% | 1.58 | 46.6% | 208 |
| meta P(win)>0.55 | +7.60% | 2.21 | 48.6% | 142 |
| meta P(win)>0.60 | +1.48% | 0.80 | 48.5% | 99 |
| meta P(win)>0.70 | +1.16% | 1.65 | 51.4% | 35 |

Every meta threshold is far worse than the simple confidence threshold, and they admit more
trades at lower win rates.

## Interpretation

The LSTM's own **confidence is already a near-optimal selector** for this edge. A secondary model
on hand-crafted features cannot improve on it (AUC 0.569 ≈ chance). This fits the project-wide
pattern: the SBER edge is a clean, simple phenomenon best captured by raw confidence; added
complexity (meta-model, more features, longer horizons, more tickers, richer architecture) does
not help and usually dilutes.

## Conclusion

- **Meta-labeling: rejected.** Keep the simple production rule (BUY conf>0.50, 3h+stop, skip weekends).
- The other #2 sub-option (ET+LSTM ensemble consensus) has a weak prior — ET's backtest was negative
  at all thresholds and meta-labeling (the stronger technique) already failed. Deprioritised.
- Stage 3 substantially closed (negative). Remaining: Stage 4 (#5 target re-engineering, #6
  representation — both low prior) and the locked SBER test-set gate.
