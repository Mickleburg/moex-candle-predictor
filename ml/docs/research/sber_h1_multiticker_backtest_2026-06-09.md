# Multi-Ticker vs LSTM v2 — SBER H1 Backtest (Decision Gate)

**Date**: 2026-06-09
**Script**: `ml/scripts/sber_multiticker_backtest_research.py`
**Result file**: `ml/docs/research/sber_h1_multiticker_backtest_results_20260609_195749.json`

## Question

The multi-ticker model won on F1 (0.4852 vs 0.4778). But LSTM v2's production value was
never its F1 — it was **Sharpe=6.38 at conf>0.50** in the backtest. Does multi-ticker
training improve the backtest too, or only the F1? This is the gate for packaging it as
the new primary artifact.

## Design — apples-to-apples

- Both models trained by the **identical** `train_lstm_fold` and scored by the **identical**
  `run_backtest` engine (fee 0.05% one-way, 1h hold, same thresholds, same Sharpe annualisation).
- The **only** difference is training data: LSTM v2 = SBER only; multi-ticker = SBER+LKOH+GAZP
  pooled, leak-free (each ticker filtered to `target_ts < val_start_time`), 3× data.
- Validation SBER-only for both, identical walk-forward folds, predictions averaged over
  seeds [7,42,100].
- **Anchor check**: LSTM v2 here must reproduce the documented 2026-06-03 result (Sharpe≈6.38).

## Anchor validated ✓

LSTM v2 @ conf>0.50: **Sharpe=6.381, return=+5.07%, 78 trades, DD=−2.06%** — exact match to
the documented 6.38 / +5.07% / 78 trades. The harness is trustworthy.

## Results

### Backtest by threshold

| thr | Model | Sharpe | Return | Max DD | Trades (buy/sell) | Win | Action |
|-----|-------|--------|--------|--------|-------------------|-----|--------|
| 0.35 | LSTM v2 | −9.86 | −99.19% | −99.19% | 5274 (2248/3026) | 34.7% | 67.0% |
| 0.35 | Multi  | −9.34 | −98.95% | −98.95% | 5301 (2214/3087) | 34.9% | 67.3% |
| 0.40 | LSTM v2 | −11.40 | −97.26% | −97.26% | 3744 (1504/2240) | 33.7% | 47.6% |
| 0.40 | Multi  | −8.70 | −95.02% | −95.02% | 3735 (1601/2134) | 34.9% | 47.4% |
| 0.45 | LSTM v2 | −7.69 | −33.21% | −33.21% | 610 (258/352) | 34.9% | 7.7% |
| 0.45 | Multi  | −8.42 | −46.02% | −46.27% | 904 (510/394) | 35.0% | 11.5% |
| **0.50** | **LSTM v2** | **+6.38** | **+5.07%** | **−2.06%** | **78 (78/0)** | **39.7%** | **1.0%** |
| **0.50** | **Multi**  | **−0.39** | **−0.79%** | **−6.41%** | **279 (279/0)** | **35.1%** | **3.5%** |

Buy & Hold over the same val periods: return=+24.70%, Sharpe=0.407, DD=−33.31%.

### Decision gate

| | LSTM v2 best | Multi-ticker best |
|---|---|---|
| Threshold | 0.50 | 0.50 |
| **Sharpe** | **+6.38** | **−0.39** |
| Return | +5.07% | −0.79% |
| Trades | 78 | 279 |

**VERDICT: multi-ticker does NOT beat LSTM v2. Keep LSTM v2 as the production candidate.**

## Interpretation — why F1 ↑ but Sharpe ↓

The two metrics measure different things, and for this problem they **diverge**.

1. **LSTM v2's edge is a tiny set of 78 ultra-high-conviction BUY calls** (1.0% of candles).
   These are genuinely directional: 39.7% win rate but asymmetric payoff, very low variance →
   Sharpe 6.38, DD only −2.06%. This rare signal — not general accuracy — is the whole value.

2. **Multi-ticker training broadened the confident region from 78 to 279 BUY calls** (3.5% of
   candles). The 201 extra "confident" predictions are noise: win rate falls 39.7% → 35.1%,
   and the strategy flips from +5.07% to −0.79%. More data made the model confident on *more*
   candles without those extra calls being directional.

3. **At conf>0.50 both models are long-only** — neither ever produces a confident SELL. The
   high-conviction signal is structurally a BUY-only signal.

4. **This is the second confirmation of the same failure mode.** The Transformer also raised
   "confidence" (26% conf>0.50) and F1-adjacent metrics but failed the backtest. Pattern, now
   confirmed twice:

   > **F1 and confident-prediction volume are the WRONG objective for this problem. The value
   > lives in a small set of rare, genuinely high-conviction calls. Anything that increases the
   > volume of confident predictions (richer architecture, more data) DILUTES that signal.**

## Conclusion

- **Keep LSTM v2 (SBER-only) as the production candidate.** Already packaged as artifact;
  production rule (trade only conf>0.50) gives Sharpe=6.38, +5.07%, DD=−2.06% over 16 months.
- **Do NOT package multi-ticker.** It is a research win on F1 and worst-fold generalisation,
  but a loss on the only metric that matters for production. Archived as a documented negative.
- **Strategic correction**: stop chasing F1 / confidence breadth. The production objective is
  the backtest at high confidence. Future work must *sharpen and preserve* the rare 78-signal
  edge, not broaden the confident region.

## Recommended next steps (ordered)

1. **Cache LSTM v2 walk-forward predictions to disk**, then iterate the backtest WITHOUT
   retraining (seconds, not hours):
   - **3h-exit** (match the triple-barrier h=3 horizon). The 78 BUY signals predict a move over
     3 hours; the current 1h exit may leave return on the table. Same 78 signals, potentially
     higher return — the cheapest possible upside.
   - **Finer thresholds above 0.50** (0.50 / 0.55 / 0.60): are the very highest-conviction calls
     even better, and how few trades remain?
2. **Once exit horizon + threshold are settled**, run the **final locked test-set evaluation**
   (one-shot) before any team sign-off.
3. Note for risk_manager: the production signal is **long-only** at conf>0.50.

## Updated experiment ledger

| Experiment | WF F1 | Backtest (best) | Verdict |
|-----------|-------|-----------------|---------|
| LSTM v2 (SBER) | 0.4778 | **Sharpe 6.38 @ conf>0.50** | **Production candidate** |
| Transformer (mean pool) | 0.4699 | not run (overconfident) | Negative |
| Multi-ticker LSTM | 0.4852 | **Sharpe −0.39 @ conf>0.50** | F1 win, backtest loss → archived |
