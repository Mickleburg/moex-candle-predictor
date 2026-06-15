# LKOH Orthogonal Refinement — oil + market/FX is the unlock — 2026-06-15

**Script**: `sber_orthogonal_research.py --ticker LKOH --groups <combo>`
**Results**: `lkoh_orthogonal_commodity_market_*`, `lkoh_orthogonal_commodity_market_sector_rates_*`.

## Ablation — forward selection over orthogonal groups

| LKOH config | Return | Sharpe | Win | Trades |
|-------------|--------|--------|-----|--------|
| OHLCV baseline | +0.50% | 0.60 | 45.0% | 56 |
| + commodity (oil/gas) | +8.57% | 2.73 | 47.3% | 129 |
| **+ commodity + market/FX** | **+11.73%** | **5.42** | **56.8%** | 81 |
| + commodity + market + sector + rates (all) | +0.17% | 0.21 | 44.0% | 50 |

## Finding — oil + market/FX unlocks LKOH; the kitchen-sink dilutes it

The winning set is **commodity (Brent) + market/FX (IMOEX, RTSI, RTSI−IMOEX spread)**:
Sharpe **5.42** (9× the OHLCV baseline's 0.60), win **56.8%** (above 50% for the first time),
+11.73% on 81 weekday trades. Economically clean: an oil producer driven by the oil price and the
ruble/market.

Adding sector+rates on top collapses it back to +0.17% — the same dilution rule seen throughout:
a targeted, parsimonious driver set works; the kitchen-sink adds noise and destroys selectivity.

## Updated ticker map

| Ticker | Status | Backtest (weekday, 3h+stop) |
|--------|--------|------------------------------|
| **SBER** | production-grade | +18.96%, Sharpe ~15 (OHLCV; orthogonal dilutes) |
| **LKOH** | **unlocked** (oil + market/FX) | **+11.73%, Sharpe 5.42, win 56.8%, 81 trades** |
| GAZP | not forecastable at 1H | no driver found (NG=Henry Hub ≠ Gazprom) |

LKOH is now a genuine second tradeable ticker — weaker than SBER (Sharpe 5.42 vs ~15) but a large
improvement over its OHLCV baseline, and win-rate-positive.

## Next steps

1. **Robustness for LKOH oil+market** (#4-style: bootstrap CI, fee stress, random baseline) to
   confirm it's not luck (81 trades).
2. **Packaging consideration (architectural):** serving LKOH predictions needs the orthogonal series
   (Brent/IMOEX/RTSI) available at INFERENCE, not just the LKOH candle_batch. Options: extend the
   input contract with a market-context block, or have the ML block fetch orthogonal data itself.
   To design before packaging the LKOH artifact (the router will pick up `research_lstm_v2_lkoh_h1`).
3. SBER: Stage 4 (#5/#6, low prior) + the locked test-set gate.
