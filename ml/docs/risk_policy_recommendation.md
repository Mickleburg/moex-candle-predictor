# ML → risk_manager: how the trading policy is delivered

**Short answer:** the per-request `ml_prediction` contract does **not** carry the trading policy.
It carries the **forecast** plus the **inputs** risk_manager needs to act. The **policy itself**
(threshold, hold, stop, weekend filter, long-only, which tickers are tradeable) is a **static,
versioned spec** published separately at [`config/ml_risk_policy.json`](../../config/ml_risk_policy.json).

## Why this split (contract vs spec)

| | Per-request `ml_prediction` contract | Static policy spec (`config/ml_risk_policy.json`) |
|---|---|---|
| Changes | every candle | rarely (versioned) |
| Owns | the FORECAST | the DECISION POLICY |
| Examples | probabilities, confidence, expected_return, signal_context (barriers, horizon), as_of | min_confidence, hold bars, stop source, skip-weekends, long-only, tradeable tickers |
| Block | ML | risk_manager (this is its recommendation) |

Reasons:
1. **Separation of concerns** — ML forecasts; risk_manager decides. The contracts already draw this
   line (`ml_prediction` = forecast; `risk_decision` = order_intent).
2. **Don't repeat static policy in every prediction** — stuffing "min_confidence=0.50, skip weekends"
   into every candle's response is noise and invites drift between producers.
3. **Single source of truth** — the threshold/stop/session policy lives in ONE place (risk_manager,
   seeded by this spec), not duplicated inside ML output.
4. **Reusability** — the same `ml_prediction` can feed different risk policies (conservative vs
   aggressive), the LLM fusion, and backtests, without ML baking in a decision.

## What the contract already gives risk_manager (so it can apply ANY policy)

- `confidence` → compare to `min_confidence` (entry threshold).
- `signal_context.lower_barrier` → absolute stop-loss level (no recomputation needed).
- `signal_context.horizon_bars` → hold horizon.
- `expected_return` → risk-adjusted edge / position sizing.
- `as_of` (Europe/Moscow) → session and weekend filtering.

risk_manager reads these per request and applies the static policy below.

## Current recommended policy (validation-grade, see spec for numbers)

- **SBER — tradeable.** BUY when `confidence > 0.50`, hold 3 bars (3h), stop at
  `signal_context.lower_barrier`, no take-profit, long-only, **skip weekend sessions**.
  Validation (weekday-only): +18.96%, Sharpe 14.95, win 73.2%, DD −1.02%, 41 trades.
  Caveats: edge decays 2023→2025; small sample; out-of-sample **test gate still pending**.
- **GAZP, LKOH — not tradeable yet.** Predictions are served, but no edge on OHLCV
  (conf>0.50 BUY win < 50%). Do not act until orthogonal data (Brent/gas/USDRUB) is added.

## Delivery mechanism

`config/ml_risk_policy.json` is the machine-readable hand-off (versioned by `spec_version`).
When risk_manager is implemented it loads this file; until then it is the documented contract of
intent. The ML block updates this file when the validated policy changes (e.g. after the test gate).
