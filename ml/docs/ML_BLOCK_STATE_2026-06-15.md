# ML Block — State & Code Review — 2026-06-15

## 1. Code-review verdict

**Status: correct and sound for a research block. Smoke tests 19/19 green. Working tree clean.**

Verified invariants (concretely, not from memory):
- **No lookahead in features** — corrupting all candles after index k leaves every feature at rows
  ≤ k byte-identical (proven for `build_per_step_features`). Volume uses `shift(1)`; orthogonal
  features use `merge_asof(direction="backward")` (verified: an evening SBER candle picks up the
  day's last index bar, never a future one).
- **Target uses the future only for LABELS** (`triple_barrier_details` reads high/low over
  [t+1, t+h]) — never for features. Labels are −1 (excluded) where the horizon runs off the end.
- **Walk-forward chronology** — `walk_forward_ranges` enforces train_end ≤ val_start; folds expand
  forward in time, no shuffling.
- **Test split untouched** — artifacts train on the first 85% (development); the last 15% is never
  read in training or model selection. The locked test gate has not been run.
- **Contract validation** — candle_batch parsing rejects duplicate/mismatched/non-finite rows;
  ml_prediction output is schema-valid; probabilities sum to 1.
- **Per-ticker routing** — request `ticker` → ticker artifact; unknown ticker → graceful
  `artifact_missing`; a ticker/artifact mismatch raises.
- **ML self-fetches orthogonal data** — LKOH served end-to-end with Brent/IMOEX/RTSI pulled by the
  block itself; the input contract is unchanged.

### Issues found (honest; none are correctness-critical)
1. **Stale metadata in the SBER artifact** — `metadata.json` still carries
   `backtest_conf050_sharpe=6.38` (the old 1h-exit number). The production rule is now 3h+stop
   (Sharpe 14.95). Informational only (downstream applies the rule), but should be refreshed.
2. **tz-handling inconsistency between trainers** — `train_lstm_artifact.py` (SBER) uses legacy
   `utc=True`; `train_lstm_orthogonal_artifact.py` (LKOH) uses `tz_aware=True` (MSK). Both yield the
   SAME wall-clock hour/dow, so no functional difference — but worth harmonising for clarity.
3. **No defensive feature-order assert at inference** — orthogonal inference rebuilds features via
   `build_combined_features(groups)` (deterministic order) and does not cross-check against the
   stored `feature_names`. Works because the builder is deterministic; a guard would be safer.
4. **`is_production=False` everywhere** — correct by invariant until the test gate passes and the
   team signs off.

## 2. How the block works

```
candle_batch JSON (ONE ticker's candles)
   → TickerModelRouter           resolves ticker → artifact dir (research_lstm_v2_<t>_h1)
   → ResearchArtifact            loads model.pt + configs (cached)
   → _lstm_feature_matrix        OHLCV/time features; if orthogonal_groups → also self-fetch
                                 Brent/IMOEX/RTSI via MarketContextProvider and build 14+K features
   → CandleLSTM                  32-step window → softmax {SELL,HOLD,BUY}
   → ml_prediction JSON          probabilities, confidence, expected_return, signal_context, diagnostics
```
Downstream (aggregator → risk_manager → execution) owns thresholds, hold, stop, sizing, portfolio.
The block only forecasts. The static trading policy is published separately in
`config/ml_risk_policy.json`.

## 3. Features (per-step, past-only)

**Base (14, all tickers)** — scale-invariant OHLCV + session:
`ret_1h, ret_3h, body, range_, upper_shadow, lower_shadow, close_pos, vol_ratio, vol_z,
hour_sin, hour_cos, dow_sin, dow_cos, ema_dist_8`. (Time features carry ~62% of the signal —
MOEX intraday seasonality.)

**Orthogonal (LKOH only, +17 → 31 total)** — self-fetched cross-instrument drivers:
- commodity (8): `BR_CONT` (Brent) + `NG_CONT` (gas) ret_1h/3h/d + vol — the genuinely-orthogonal
  oil/gas price (continuous front-month FORTS futures).
- market (9): `IMOEX`, `RTSI` ret_1h/3h/d + vol, and `rtsi_imoex_spread_ret` (= implicit USD/RUB,
  since USD-spot was suspended mid-2024).

## 4. Target — triple-barrier

For candle t: `past_vol = std(returns[t-12:t])`; `upper = close·(1+1.25·vol)`,
`lower = close·(1−1.25·vol)`. Look forward [t+1, t+3]: first barrier hit → BUY (upper) / SELL
(lower); timeout → HOLD. Confirmed optimal vs grids (h3/k1.25/w12). Class balance ≈ 33/34/33.

## 5. Model

`CandleLSTM`: LSTM(input=14 or 31, hidden=128, layers=2, dropout=0.3) → last hidden →
Linear(128→64)→ReLU→Dropout→Linear(64→3). 32-step sliding window. CrossEntropy, Adam(lr=1e-3,
wd=1e-4) + CosineAnnealing, early stopping. Seed-averaged over [7,42,100] for research; single seed
(42) for the packaged artifact (trained on 85%).

## 6. Results

| Ticker | Model | WF macro-F1 | Backtest (val, 3h+stop, weekday) | Robustness | Tradeable |
|--------|-------|-------------|----------------------------------|------------|-----------|
| **SBER** | OHLCV LSTM v2 | 0.4778 | **+18.96%, Sharpe 14.95, win 73.2%, 41 trades, DD −1.02%** | bootstrap P(profit)=100%, beats 100% random, fee-OK ~0.20% | **yes** |
| **LKOH** | oil+market LSTM | ~0.46 | **+11.73%, Sharpe 5.42, win 56.8%, 81 trades** | P(profit)=97.3%, beats 100% random, fee-OK ~0.10% | **yes (weaker)** |
| GAZP | — | ~0.45 | no edge (win <50%) | — | no (served, not traded) |

**Production rule (both):** BUY when confidence > 0.50; hold 3h with a stop at the lower volatility
barrier; no take-profit; long-only; skip weekend sessions.

### What did NOT work (7 negative results — the meta-lesson)
multi-ticker joint training, transformer (mean-pool), multi-horizon (h6/h12), meta-labeling,
orthogonal-on-SBER, asym/seq-len target variants, GAZP (any route). **Pattern: the edge is a rare,
clean, high-conviction signal; broadening it (more data, richer architecture, more features,
longer horizons) dilutes it. Simple confidence is a near-optimal selector.**

### Key findings
- Matching the exit horizon to the target (3h) tripled return (+5.07%→+16.08%); stop-loss added
  return + halved drawdown; take-profit hurts (caps momentum winners).
- The edge concentrates in Friday + evening session (MSK) + high volatility; it decays 2023→2025.
- LKOH's oil driver works because it is genuinely exogenous; GAZP's gas (Henry Hub) is the wrong
  driver (Gazprom = European gas + geopolitics).

## 7. Current status

- **All research closed** — 4 stages + 6 directions + orthogonal track done.
- **Production config settled**; two tradeable tickers (SBER, LKOH) packaged & served.
- **Infrastructure ready**: router, extended contract, risk-policy spec, self-fetch provider,
  tz hygiene, 19/19 smoke tests.
- **`is_production=False`** — pending the one remaining step.
- **Only remaining step: the locked one-shot TEST-SET gate** (SBER + LKOH, last 15% ~2026,
  irreversible). Not yet run — awaiting explicit go-ahead. Carry the 2025-decay caveat.

### Recommended small cleanups before the gate (optional)
Refresh stale SBER artifact metadata; harmonise trainer tz handling; add a feature-order guard at
orthogonal inference. None block the gate.
