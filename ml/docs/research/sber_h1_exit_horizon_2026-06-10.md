# LSTM v2 Exit-Horizon + Fine-Threshold Backtest — SBER H1

**Date**: 2026-06-10
**Script**: `ml/scripts/sber_lstm_exit_horizon_backtest.py`
**Result file**: `ml/docs/research/sber_h1_exit_horizon_results_20260610_123037.json`
**Prediction cache**: `ml/artifacts/lstm_v2_wf_predictions.npz` (gitignored, reusable)

## Question

LSTM v2's documented edge (Sharpe 6.38, +5.07%) used a **1-hour exit**, but the triple-barrier
target is defined over **3 hours** — a BUY signal predicts a move within [t+1, t+3]. Exiting at
t+1 may close the trade before the predicted move completes. Does matching the exit to the target
horizon (and tightening the confidence threshold) improve the SAME signals — with no retraining?

## Method

LSTM v2 walk-forward predictions (proba + candle idx, seed-averaged over [7,42,100]) collected
once and cached. A generalized engine then backtests every (hold ∈ {1,2,3}h) × (threshold ∈
{0.45,0.50,0.55,0.60}) combination in seconds. Cooldown: no overlapping positions (realisable
single-account equity). Sharpe annualised by `sqrt(HOURS_PER_YEAR / hold_h)`; at hold=1 this
reproduces the original engine. **Anchor: hold=1h thr=0.50 → Sharpe=6.381 ✓ (documented 6.38).**
Fee 0.05% one-way. Validation periods only (test split untouched).

## Results

| hold | thr | Sharpe | Return | Avg/trade | Max DD | Win | Trades |
|------|-----|--------|--------|-----------|--------|-----|--------|
| 1h | 0.45 | −7.69 | −33.21% | −0.066% | −33.21% | 34.9% | 610 |
| 1h | 0.50 | 6.38 | +5.07% | +0.064% | −2.06% | 39.7% | 78 |
| 1h | 0.55 | 14.33 | +4.31% | +0.137% | −0.71% | 45.2% | 31 |
| 1h | 0.60 | 22.30 | +3.54% | +0.233% | −0.29% | 46.7% | 15 |
| 2h | 0.50 | 9.23 | +12.04% | +0.213% | −1.74% | 63.0% | 54 |
| 2h | 0.60 | 25.48 | +6.31% | +0.685% | −0.12% | 88.9% | 9 |
| **3h** | **0.50** | **9.56** | **+16.08%** | **+0.308%** | **−2.16%** | **71.4%** | **49** |
| 3h | 0.55 | 11.25 | +8.57% | +0.491% | −1.40% | 70.6% | 17 |
| 3h | 0.60 | 26.17 | +9.91% | +1.060% | 0.00% | 100.0% | 9 |

(1h/0.45, 2h/0.45, 3h/0.45 all negative — below conf>0.50 the signal is noise, as before.)

Buy & Hold over the same periods: +24.70%, Sharpe 0.407, **DD −33.31%**.

## Key findings

### 1. Matching exit to the 3h target horizon roughly triples return on the same signals

At conf>0.50, going from a 1h to a 3h exit:
- Return **+5.07% → +16.08%** (3.2×)
- Win rate **39.7% → 71.4%**
- Sharpe **6.38 → 9.56**
- Drawdown stays tiny (−2.06% → −2.16%)
- Trades 78 → 49 (a few are merged by the no-overlap cooldown)

The 1h exit was measuring the signal at the wrong horizon. The triple-barrier label predicts a
move *within 3 hours*; closing at 1h cut trades before the move completed, which is why the 1h
win rate looked near-random (39.7%). At the horizon the model was actually trained to predict,
the signal is **71.4% accurate**. This is a measurement fix, not a model change.

### 2. The trend is monotonic and sensible — evidence of real signal

Across the grid, higher confidence → fewer trades but higher win rate and higher avg return per
trade (0.06% → 0.23% at 1h; 0.31% → 1.06% at 3h). Longer hold (up to the 3h barrier) → higher
return. Both gradients point the same way and are smooth. Noise would not produce this structure.

### 3. Production sweet spot: 3h hold, conf>0.50

- **+16.08% return, Sharpe 9.56, win 71.4%, 49 trades, max DD −2.16%** over ~16 months of val.
- Beats Buy & Hold's +24.70% on **risk-adjusted** terms by a wide margin (DD −2.16% vs −33.31%,
  Sharpe 9.56 vs 0.41) while deploying capital only ~1% of the time.
- 49 trades is a statistically reasonable sample (unlike the 9-trade/100%-win 3h/0.60 corner,
  which is too small to trust — reported but not selected).

### 4. Still long-only at high conviction

Every conf>0.50 trade is a BUY (n_sell=0 at all thresholds ≥0.50). The model never produces a
confident SELL. The production signal is structurally long-only — a note for risk_manager.

## Conclusion

**The production rule is upgraded: trade BUY when LSTM v2 confidence > 0.50, hold 3 hours
(exit at t+3).** This matches the triple-barrier target horizon and lifts the validated edge from
+5.07% to **+16.08%** (Sharpe 9.56, win 71.4%, DD −2.16%) — with zero model changes, just a
corrected exit. This materially strengthens the production case for LSTM v2.

### Next step — final locked test-set evaluation (the last gate)

The config (3h hold, conf>0.50) is now settled on validation. The remaining gate is the **one-shot
test-set evaluation** (last 15%, ~3760 candles, never touched): train up to test_start, predict on
test, apply the 3h/conf>0.50 rule, report return/Sharpe/win. Per project invariant #1 this is done
exactly once. If it holds out-of-sample, LSTM v2 is ready for team sign-off.

## Updated production checklist impact

- [x] Positive, **strong** backtest: 3h/conf>0.50 → Sharpe 9.56, +16.08%, DD −2.16% (val)
- [x] Exit horizon matched to target (3h)
- [ ] **Final locked test-set evaluation** ← next
- [ ] Team sign-off (threshold 0.50, hold 3h, long-only)
