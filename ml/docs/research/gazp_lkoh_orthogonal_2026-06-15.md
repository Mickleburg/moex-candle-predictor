# Orthogonal Features — GAZP & LKOH (all groups) — 2026-06-15

**Scripts**: `sber_orthogonal_research.py --ticker {GAZP,LKOH} --groups market,sector,rates`
**Results**: `gazp_orthogonal_*_20260615_001202.json`, `lkoh_orthogonal_*_20260615_003657.json`

## Result — orthogonal index features do NOT unlock GAZP/LKOH

| Ticker | Config | WF F1 | Return | Sharpe | Win | Trades |
|--------|--------|-------|--------|--------|-----|--------|
| GAZP | OHLCV | 0.4521 | +0.75% | 0.29 | 41.0% | 131 |
| GAZP | + orthogonal (18) | 0.4527 | **−10.65%** | −1.63 | 38.8% | 224 |
| LKOH | OHLCV | 0.4567 | +0.50% | 0.60 | 45.0% | 56 |
| LKOH | + orthogonal (18) | 0.4509 | **−1.64%** | −1.12 | 45.2% | 62 |

Worse on the backtest for both: win stays below 50% (no edge) and the 18-feature set dilutes
(GAZP 131→224 trades, all noise). The index-based orthogonal set did not create a tradeable signal.

## Key caveat — the sector index is NOT truly orthogonal

`MOEXOG` is the **oil & gas sector index** — a basket of LKOH/GAZP/ROSN/etc. It is highly
collinear with the very stocks we're predicting, so it carries little genuinely new information.
The economically orthogonal driver for oil/gas equities is the **underlying commodity** — Brent
crude and natural gas — which was **deferred** (FORTS futures need front-month roll stitching).
So this experiment tested macro-index context, NOT the strongest hypothesised driver (oil price).

Conclusion precisely:
- **Orthogonal INDEX features (market/sector/rates): negative** for all three tickers (dilute SBER,
  don't unlock GAZP/LKOH).
- **Orthogonal COMMODITY features (Brent/gas futures): untested** — the one genuinely-orthogonal,
  economically-strongest signal for LKOH/GAZP remains open.

## Cumulative picture (many experiments)

GAZP/LKOH resist 1H modelling by every route tried: OHLCV (no edge), multi-ticker joint (dilutes),
and now macro-index orthogonal (no help). Their moves appear exogenous/jump-like at 1H. SBER
remains the only production-grade ticker; its OHLCV edge is strong and is only diluted by additions.

## Decision fork

1. **Final orthogonal attempt — Brent/gas futures for LKOH/GAZP.** Build continuous front-month
   BR (oil) and NG (gas) series from FORTS, add as features, retrain. The strongest economic
   hypothesis, genuinely orthogonal. Cost: roll-stitching code + ~compute. Prior is now weak given
   how consistently GAZP/LKOH resist 1H prediction, but it is the clean way to exhaust the thesis.
2. **Accept SBER-only and finish the SBER plan.** Ship the one thing that robustly works: proceed
   to Stage 3 (#2 meta-labeling/ensemble for SBER) and Stage 4 (#5/#6), then the locked SBER
   test-set gate. Treat Brent futures as an optional later follow-up.
