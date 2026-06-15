# Orthogonal Features — SBER gate (all groups) — 2026-06-15

**Script**: `ml/scripts/sber_orthogonal_research.py --ticker SBER --groups market,sector,rates`
**Result**: `ml/docs/research/sber_orthogonal_market_sector_rates_20260614_235121.json`

## Result — orthogonal does NOT help SBER (all-groups)

| Config | WF F1 | Return | Sharpe | Win | Trades |
|--------|-------|--------|--------|-----|--------|
| OHLCV baseline | 0.4814 | **+18.96%** | **14.95** | 73.2% | 41 |
| + market+sector+rates (18 feats) | 0.4730 | +17.72% | 11.02 | 65.3% | 49 |
| **Delta** | −0.008 | **−1.24%** | **−3.93** | −7.9pp | — |

Adding all 18 orthogonal features slightly **hurts** SBER on every metric (lower return, Sharpe,
win rate; more trades). Same dilution pattern: more inputs broaden the confident set with some
noise. SBER's OHLCV edge is already strong (Sharpe 14.95) and well-selected; the kitchen-sink
dilutes it.

## Interpretation & decision

- A **single** group (e.g. rates/RGBI, the hypothesised bank driver) might still help where the
  full set dilutes — that's an open ablation. But SBER **already works well**; chasing a marginal
  SBER gain is low-EV.
- The orthogonal-data thesis was always strongest for **GAZP/LKOH**, which had **no** OHLCV edge
  (conf>0.50 BUY win 41%/45% < 50%) because their moves are exogenous (oil/gas/FX). There is no
  precious rare signal there to dilute — orthogonal information can only help or be neutral.

**Decision:** pivot to GAZP/LKOH with orthogonal features (the real unlock test). SBER stays on its
OHLCV production rule; optional SBER single-group ablation deferred (low EV).

## Status
- SBER orthogonal (all groups): negative (−1.24% return, −3.93 Sharpe).
- Next: GAZP + LKOH with groups=market,sector,rates (oil&gas sector, FX spread, market, rates).
