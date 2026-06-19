# SBER Multi-Horizon — Stage 2 (#1, multi-horizon half) — 2026-06-14

**Script**: `ml/scripts/sber_multihorizon_research.py`
**Result**: `ml/docs/research/sber_multihorizon_results_20260614_224401.json`

## Hypothesis

The SBER edge is rare (~41 weekday trades / 16 mo at h=3). Different triple-barrier horizons
(h=6, h=12) give different entry signals on the same 1H series — maybe more independent
high-conviction opportunities of similar quality. Same recipe per horizon (only barrier_horizon
changes), judged by the production backtest (BUY conf>0.50, hold h + lower-barrier stop, skip
weekends).

## Results

| Horizon | WF F1 | Trades | Return | Sharpe | Win |
|---------|-------|--------|--------|--------|-----|
| **h=3 (baseline)** | 0.4814 | 41 | **+18.96%** | **14.95** | 73.2% |
| h=6 | 0.4041 | 413 | −4.57% | −0.30 | 37.3% |
| h=12 | 0.4040 | 536 | −21.59% | −0.87 | 28.7% |
| combined (pooled, cooldown) | — | 626 | −30.44% | −1.31 | 29.6% |
| | | | | composition: h12=411, h6=206, h3=9 |

## Finding — NEGATIVE. Longer horizons destroy selectivity.

At h≥6 the HOLD class collapses (a barrier is almost always hit within 6–12h; label balance
HOLD 17% at h6, 9% at h12). The model becomes "confident" almost everywhere, so the conf>0.50
filter no longer selects a rare high-quality set — it admits hundreds of low-quality trades with
win rates **below random** (37%, 29%). F1 also drops to ~0.40. Combining horizons is dominated by
the bad h12/h6 entries (411+206 of 626) → −30.44%.

This is the **same dilution failure** seen with the transformer (overconfident) and multi-ticker
joint training (broadened the confident set with noise). Confirmed meta-lesson:

> The SBER edge is a NARROW, SPECIFIC phenomenon — rare high-conviction 1H/h=3 BUY signals.
> Broadening it (more data, richer architecture, longer horizons) dilutes the selectivity that
> makes it work. Opportunity expansion via these routes does not work.

## Conclusion & next

- **Multi-horizon: rejected.** h=3 remains the only tradeable horizon.
- The other half of #1, **multi-timeframe (15m/Daily)**, is genuinely different (finer/coarser bars,
  not a horizon stretch on the same bars) and could have its own rare-high-conviction structure.
  But given the consistent dilution prior, it is **deprioritised** below the orthogonal-data track
  (genuinely new information vs more of the same). Revisit 15m only to exhaust #1 if desired.
- **Proceed to the orthogonal-data gate** (higher EV): orthogonal features add information the
  candles cannot contain, the only route shown to plausibly help — especially for GAZP/LKOH.
