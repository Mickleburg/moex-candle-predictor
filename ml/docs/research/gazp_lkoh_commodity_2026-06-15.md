# Orthogonal Commodity (Brent/gas) — GAZP & LKOH — 2026-06-15

**Script**: `sber_orthogonal_research.py --ticker {LKOH,GAZP} --groups commodity`
**Continuous futures**: `BR_CONT` (Brent), `NG_CONT` (gas), built by `download_futures_continuous.py`.
**Results**: `lkoh_orthogonal_commodity_*`, `gazp_orthogonal_commodity_*`.

## Result — the first POSITIVE orthogonal signal (LKOH ← oil)

| Ticker | Config | WF F1 | Return | Sharpe | Win | Trades |
|--------|--------|-------|--------|--------|-----|--------|
| LKOH | OHLCV baseline | 0.4567 | +0.50% | 0.60 | 45.0% | 56 |
| **LKOH | + commodity (Brent/gas)** | 0.4620 | **+8.57%** | **2.73** | 47.3% | 129 |
| GAZP | OHLCV baseline | 0.4521 | +0.75% | 0.29 | 41.0% | — |
| GAZP | + commodity (Brent/gas) | 0.4553 | −2.37% | −0.49 | 42.2% | 161 |

## Findings

### LKOH: oil works — the orthogonal thesis is confirmed

Brent lifts LKOH from Sharpe 0.60 → **2.73** and return +0.50% → **+8.57%**. This is the first time
orthogonal data helped, and it validates the whole approach: the **genuinely-orthogonal** driver
(the actual oil price) carries information the candles cannot — unlike the collinear MOEXOG sector
index (which earlier gave −1.64%). Economically obvious: Lukoil is an oil producer.

**Caveat:** win rate is still 47.3% (<50%); the positive return comes from payoff asymmetry
(stop cuts losers small, winners run to 3h). Sharpe 2.73 is modest vs SBER's 14.95. So this is a
**partial unlock / promising direction**, not yet production-grade. Refinement underway
(oil+market/FX, oil+full).

### GAZP: gas (Henry Hub) is the wrong driver

NG on MOEX FORTS tracks US Henry Hub gas, but Gazprom is driven by **European** gas dynamics and,
post-2022, geopolitics/sanctions — none of which is in any tradeable price series. Brent didn't
help either. GAZP remains unpredictable at 1H by every route tried (OHLCV, multi-ticker, index,
commodity). Likely genuinely unforecastable from price data at this horizon.

## Updated picture

- **SBER**: production-grade (OHLCV, Sharpe ~15). Orthogonal/complexity only dilutes it.
- **LKOH**: partially unlocked by oil (Sharpe 0.60→2.73). Worth refining; not yet production-grade.
- **GAZP**: not forecastable at 1H; no driver found. Stop here.

## Next
- Refine LKOH (oil + market/FX; oil + full set) — running.
- If LKOH reaches a robust positive edge → robustness (#4-style) + package artifact (router ready).
- SBER: Stage 4 (#5/#6, low prior) + the locked test-set gate.
