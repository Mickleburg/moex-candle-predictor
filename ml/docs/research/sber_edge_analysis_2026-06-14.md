# SBER Edge Analysis — Stage 1 (#3 where the edge lives + #4 robustness) — 2026-06-14

**Script**: `ml/scripts/sber_edge_analysis.py`
**Result file**: `ml/docs/research/sber_edge_analysis_results_20260614_222722.json`
**Input**: cached LSTM v2 walk-forward predictions. No retraining (7s).
**Rule analysed**: BUY conf>0.50, hold 3h + lower-barrier stop, no take-profit (50 trades, val).

> **Timezone note**: the SBER `begin` timestamps are MSK values labelled as UTC (tz-localised,
> not converted). This is INTERNALLY CONSISTENT (the model learned its hour/dow features on these
> same values), so it is not a model bug — but "hour 22" below means 22:00 MSK (evening session).
> Worth a data-hygiene fix later; does not affect conclusions.

## #3 — Where the edge lives

### By weekday — the edge is overwhelmingly Friday
| dow | n | win | total ret |
|-----|---|-----|-----------|
| Mon | 6 | 66.7% | +0.94% |
| **Fri (4)** | **33** | **75.8%** | **+17.87%** |
| Sat (5) | 2 | 100% | +0.55% |
| **Sun (6)** | **7** | **14.3%** | **−1.73%** |

Friday alone carries essentially the entire edge. Sat/Sun are MOEX's **new 2025 weekend sessions**
(thin liquidity) — the Sunday cluster is the only consistently losing group.

### By hour (MSK) — the edge is the evening session
Hours 22 (n=27, +11.97%) and 23 (n=6, +3.54%) dominate; midday hour 12 (n=7, −1.73%, all the Sunday
weekend trades) is the loser. The pre-weekend Friday evening session is the sweet spot.

### By volatility — stronger in high vol
High-vol tercile: n=17, win **82.4%**, +9.55%. Low-vol: win 47.1%, +2.98%. The momentum edge is
sharper when volatility is elevated.

### By year — the edge decays
| year | n | win | total ret | mean/trade |
|------|---|-----|-----------|------------|
| 2023 | 7 | 100% | +5.33% | +0.747% |
| 2024 | 27 | 66.7% | +9.33% | +0.334% |
| **2025** | 16 | **50.0%** | +2.07% | +0.130% |

2025 is the weakest (50% win, barely positive). Partly the new weekend sessions, partly possible
regime drift. **This is a flag for the test gate** — the test set (2026) is the most recent period.

### Confidence buckets (noisy at small n)
0.50–0.55: n=35, win 68.6%. 0.55–0.60: n=12, win 50%. 0.60+: n=3, win 100%. No clean monotonic
trend at trade level (small cells); the cumulative threshold sweep earlier is the more reliable view.

## #4 — Robustness (the edge is real, not luck)

### Bootstrap (B=20000 resamples of the 50 trade returns)
- Total return: p05 **+8.75%**, p50 +17.34%, p95 +27.56%, **P(profit) = 100%**.
- Sharpe: p05 **7.27**, p50 11.97, p95 16.76, **P(>0) = 100%**.
- Even the 5th percentile is strongly positive — not a couple of lucky trades.

### Fee stress (one-way)
| fee | return | Sharpe |
|-----|--------|--------|
| 0.05% (assumed) | +17.54% | 11.85 |
| 0.10% | +11.83% | 8.22 |
| 0.15% | +6.38% | 4.58 |
| 0.20% | +1.20% | 0.95 |
| 0.30% | −8.44% | −6.31 |

Break-even ≈ 0.20–0.22% one-way. MOEX retail fees are ~0.05% → **~4× margin**. Robust to realistic costs.

### Random-selection baseline (model selection IS the value)
Going long every val candle (3h+stop) averages **−0.088%/trade** — randomly trading SBER in this
period LOSES. The model's selected trades average **+0.326%/trade**, beating **100%** of random
50-trade picks. The edge is genuine selection skill, not "SBER just went up".

## Conclusions

1. **The edge is statistically real**: bootstrap P(profit)=100% (p05 +8.75%), survives ~4× realistic
   fees, and beats 100% of random selections. High confidence it is not luck.
2. **It is concentrated**: Friday + evening session (MSK) + high volatility. This is both an
   opportunity (filter) and a concentration risk.
3. **It decays over time** (2025 weakest) — the main caveat for the upcoming test-set gate.

### Concrete, principled filter → hand to risk_manager
**Exclude MOEX weekend sessions (Sat/Sun).** They are a thin, new-in-2025 regime where the edge
breaks (Sunday cluster −1.73%). Effect on the rule:

| | trades | win | return | Sharpe |
|---|--------|-----|--------|--------|
| with weekend | 50 | 66.0% | +17.54% | 11.85 |
| **weekday-only** | 41 | **73.2%** | **+18.96%** | **14.95** |

This is a trading-session policy (WHEN to act), so it belongs in **risk_manager**, not the ML block —
consistent with the architecture (ML forecasts; risk_manager filters sessions). The ML model is
unchanged; we just recommend the downstream not act on weekend sessions.

## Stage-1 verdict

The SBER edge passes robustness and is well-characterised. Recommended spec for risk_manager:
**BUY conf>0.50, hold 3h + lower-barrier stop, no take-profit, long-only, skip weekend sessions.**
Carry the temporal-decay caveat into the test gate. Proceed to Stage 2 (#1: multi-timeframe /
multi-horizon) to expand the opportunity set.
