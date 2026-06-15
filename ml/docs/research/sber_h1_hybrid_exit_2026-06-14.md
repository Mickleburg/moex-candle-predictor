# LSTM v2 — Exit-Design 2×2 Matrix (Hybrid Exits) — SBER H1

**Date**: 2026-06-14
**Script**: `ml/scripts/sber_lstm_hybrid_exit.py`
**Result file**: `ml/docs/research/sber_h1_hybrid_exit_results_20260614_124732.json`
**Inputs**: cached LSTM v2 predictions + `triple_barrier_details` barriers. No retraining (3s).

## Question

The fixed-3h exit (+16.08%) beat the full triple-barrier exit (+7.10%) because the take-profit
caps momentum winners; but full-TB had a lower drawdown (−1.15% vs −2.16%) thanks to its stop-loss.
This completes the exit-design 2×2 — {take-profit on/off} × {stop-loss on/off} — to find the best
combination. All on the SAME BUY/conf>0.50 signals.

## Results (BUY, conf>0.50, horizon 3h)

| Exit design | Return | Sharpe | Win | Avg/trade | Max DD | Trades | Mean hold |
|-------------|--------|--------|-----|-----------|--------|--------|-----------|
| fixed 3h (TP off, stop off) | +16.08% | 9.56 | 71.4% | +0.308% | −2.16% | 49 | 3.00h |
| **hybrid A: stop only (TP off, stop ON)** | **+17.54%** | **11.85** | 66.0% | +0.326% | **−1.02%** | 50 | 2.74h |
| hybrid B: take-profit only (TP ON, stop off) | +8.14% | 7.76 | 78.4% | +0.155% | −1.40% | 51 | 2.47h |
| full TB (TP ON, stop ON) | +7.10% | 7.68 | 69.2% | +0.133% | −1.15% | 52 | 2.25h |

### 2×2 (total return / max drawdown)

| | stop OFF | stop ON |
|------|----------|---------|
| **TP OFF** | +16.08% / −2.16% | **+17.54% / −1.02%** ⭐ |
| **TP ON** | +8.14% / −1.40% | +7.10% / −1.15% |

## Findings

### Stop-loss helps; take-profit hurts — strongly and consistently

- **Stop-loss ON improves both axes** (TP-off column): return +16.08% → +17.54%, drawdown
  −2.16% → **−1.02%** (halved). The lower-barrier stop truncates the worst losers; winners never
  touch the lower barrier, so they are untouched.
- **Take-profit is destructive** (entire TP-ON row collapses to ~7–8%). Capping winners at the
  upper barrier kills the momentum edge — the high-conviction BUY signals routinely run past it.

### Best design: hold 3h, stop-loss at the lower barrier, NO take-profit (hybrid A)

**+17.54% return, Sharpe 11.85, win 66.0%, max DD −1.02%, return/DD ≈ 17×.** Strictly dominates the
previous fixed-3h rule: higher return AND half the drawdown. This is classic asymmetric trade
management — *let winners run, cut losers* — and here it improves every risk metric at once.

Win rate dips (71.4% → 66.0%) because the stop converts a few would-have-recovered dips into small
losses, but avg return per trade rises (+0.308% → +0.326%) and the worst-case drawdown halves: the
stop trades a little hit-rate for a lot of tail protection. Net positive on return, Sharpe, and DD.

### Caveat

The stop assumes execution at the lower barrier when the candle low touches it (a stop order),
modulo slippage. Validation-set only; the locked test-set evaluation remains the final gate.

## Conclusion — production rule upgraded

**BUY when LSTM v2 confidence > 0.50; hold up to 3h with a stop-loss at the lower volatility
barrier (close × (1 − 1.25·past_vol)); no take-profit; long-only.**

Backtest (val): **+17.54%, Sharpe 11.85, win 66.0%, max DD −1.02%, 50 trades.** Strictly better
than the prior fixed-3h rule on return, Sharpe, and drawdown.

## Exit-design ledger

| Rule | Return | Sharpe | Max DD | Status |
|------|--------|--------|--------|--------|
| 1h fixed | +5.07% | 6.38 | −2.06% | original |
| 3h fixed | +16.08% | 9.56 | −2.16% | superseded |
| full triple-barrier | +7.10% | 7.68 | −1.15% | rejected (TP caps winners) |
| take-profit only | +8.14% | 7.76 | −1.40% | rejected (TP caps winners) |
| **3h + stop-loss only** | **+17.54%** | **11.85** | **−1.02%** | **production rule** |
