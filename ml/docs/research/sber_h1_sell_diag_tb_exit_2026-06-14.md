# LSTM v2 — SELL Diagnosis + True Triple-Barrier Exit — SBER H1

**Date**: 2026-06-14
**Script**: `ml/scripts/sber_lstm_sell_diag_tb_exit.py`
**Result file**: `ml/docs/research/sber_h1_sell_diag_tb_exit_results_20260614_124152.json`
**Inputs**: cached LSTM v2 walk-forward predictions (`ml/artifacts/lstm_v2_wf_predictions.npz`),
triple-barrier details from `src.nlp.targets.triple_barrier_details` (same code that makes labels).
No retraining — pure analysis, 4 seconds.

---

## #5 — Why does a confident SELL never fire? (answered)

### (a) Confidence ceiling per predicted class

| argmax class | count | share | mean conf | **max conf** | >0.45 | >0.50 | >0.55 |
|--------------|-------|-------|-----------|--------------|-------|-------|-------|
| SELL | 3058 | 38.8% | 0.417 | **0.485** | 352 | **0** | 0 |
| HOLD | 2524 | 32.1% | 0.595 | 0.963 | 2072 | 1804 | 1553 |
| BUY  | 2290 | 29.1% | 0.415 | 0.697 | 258 | 78 | 31 |

The model **leans SELL more often than BUY** (38.8% vs 29.1% argmax), but SELL confidence is
**structurally capped below 0.50** (max 0.485) — not a single SELL prediction clears the trading
threshold. BUY does (78 above 0.50).

### (b) Market fact — realized val-period outcomes are balanced

| Realized triple-barrier label (val) | count | share |
|-------------------------------------|-------|-------|
| SELL (lower-first) | 2640 | 33.5% |
| HOLD (timeout) | 2692 | 34.2% |
| BUY (upper-first) | 2540 | 32.3% |

The validation period had **as many down-moves as up-moves**. The dead SELL side is **not** a
bull-market artifact — the market produced plenty of SELL outcomes; the model just can't call them.

### (c) Latent SELL edge — none

Shorting every SELL-leaning candle loses badly at every confidence level:

| filter | n | fixed-3h return | win | triple-barrier return | win |
|--------|---|-----------------|-----|-----------------------|-----|
| SELL conf>0.00 | 3058 | −93.27% | 43.2% | −95.56% | 46.8% |
| SELL conf>0.40 | 2240 | −89.22% | 42.3% | −89.17% | 47.6% |
| SELL conf>0.45 | 352 | −34.02% | 39.5% | −34.54% | 46.9% |

Win rate stays below 50% and returns are deeply negative. There is **no directional edge** in the
SELL leanings — they are uninformative.

### Diagnosis

The LSTM **genuinely cannot predict down-moves** on MOEX 1H from OHLCV+time features. This is the
classic equity asymmetry — *"up the stairs, down the elevator"*: declines are jump-like, driven by
exogenous shocks (news, macro) absent from candle history, while high-conviction up-moves are
momentum/continuation patterns the LSTM can read. The softmax reflects this honestly by never
concentrating probability mass on SELL.

**Consequence**: long-only at conf>0.50 is **correct by design, not a defect**. The direction
"fix the SELL side / add shorts" is **closed** — proven to have no edge. Do not pursue it.

---

## #1 — True triple-barrier exit vs fixed exits (BUY, conf>0.50)

| Exit | Return | Sharpe | Win | Avg/trade | Max DD | Trades | Mean hold |
|------|--------|--------|-----|-----------|--------|--------|-----------|
| 1h (fixed) | +5.07% | 6.38 | 39.7% | +0.064% | −2.06% | 78 | 1.00h |
| **3h (fixed)** | **+16.08%** | **9.56** | 71.4% | +0.308% | −2.16% | 49 | 3.00h |
| TB (barriers) | +7.10% | 7.68 | 69.2% | +0.133% | **−1.15%** | 52 | 2.25h |

TB exit outcome breakdown: 30 win_barrier, 10 timeout, 8 stop, 4 stop_ambiguous.

### Finding — fixed 3h beats true triple-barrier execution

Counterintuitive but clear: the take-profit at the upper barrier (~1.25×vol) **caps winners**. The
high-conviction BUY signals are strong momentum — price frequently **runs past the upper barrier**
within the 3h window (30 of 52 trades touched it). Holding to t+3 captures that extra upside
(avg/trade +0.308%); taking profit at the barrier truncates it (+0.133%). Hence fixed-3h's +16.08%
vs TB's +7.10%.

TB's only advantage is a **lower drawdown** (−1.15% vs −2.16%) — its stop-loss cuts the worst
losers. This points to a hybrid worth testing (below), but on return and Sharpe, **fixed 3h wins
and remains the production rule**.

---

## Conclusions

1. **SELL is dead for a real, unfixable reason** — no edge exists (shorting loses −93%), confirmed
   on a balanced market. Long-only is correct. Close this direction.
2. **Fixed 3h exit stays the production rule** (+16.08%, Sharpe 9.56). True triple-barrier execution
   is worse because it caps momentum winners.
3. Production rule unchanged: **BUY when conf>0.50, hold 3h, long-only.**

## Cheap follow-up worth testing next

**Hybrid exit: hold 3h but with a lower-barrier stop-loss only (no upper take-profit cap).** This
keeps fixed-3h's upside (let momentum winners run to t+3) while cutting the worst losers (stop at
the lower barrier), aiming for ~+16% return with TB's smaller drawdown. One more cached-prediction
backtest, no retraining.
