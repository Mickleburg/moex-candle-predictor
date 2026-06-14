# Per-Ticker LSTM Validation — Does the SBER Recipe Transfer? — 2026-06-14

**Script**: `ml/scripts/per_ticker_lstm_research.py`
**Result file**: `ml/docs/research/per_ticker_lstm_results_20260614_130848.json`
**Caches**: `ml/artifacts/lstm_v2_wf_predictions_{gazp,lkoh}.npz` (SBER reused existing cache).

## Question

The ML block serves `ml_prediction` per ticker; downstream builds the portfolio. So we need a
good model PER TICKER. SBER is validated (+17.54%, Sharpe 11.85). Does the identical recipe
(CandleLSTM v2, 14 features, 4-fold walk-forward, seeds [7,42,100]) give tradeable edge on GAZP
and LKOH? Method is identical — only the ticker changes.

## Results

| Ticker | WF F1 | SELL/HOLD/BUY F1 | 3h+stop: return | Sharpe | win | DD | trades |
|--------|-------|------------------|-----------------|--------|-----|-----|--------|
| **SBER** | 0.4814 | 0.465/0.584/0.395 | **+17.54%** | **11.85** | **66.0%** | −1.02% | 50 |
| GAZP | 0.4521 | 0.492/0.545/**0.319** | +0.75% | 0.29 | **41.2%** | −6.55% | 131 |
| LKOH | 0.4567 | 0.473/0.543/**0.353** | +0.50% | 0.60 | **44.6%** | −2.14% | 56 |

Fixed-3h (no stop) for reference: SBER +16.08%, **GAZP −1.49%, LKOH −5.32%** (both negative).

## Findings

### F1 transfers; tradeable edge does NOT

WF macro-F1 lands in the same 0.45–0.48 band for all three — the model predicts comparably well
everywhere. But the high-conviction BUY edge that makes SBER valuable **does not transfer**:

- **GAZP/LKOH win rates at conf>0.50 are below 50%** (41.2%, 44.6%) — no directional edge. SBER is 66%.
- **BUY-class F1 is markedly weaker** on GAZP (0.319) and LKOH (0.353) vs SBER (0.395) — the models
  are worst exactly on the class we trade.
- GAZP fires **131** "confident" trades (vs SBER's 50) yet is wrong more often — overconfident and
  uninformative, the same dilution pattern seen with the transformer and multi-ticker joint training.
- The stop-loss only rescues GAZP/LKOH from negative (fixed-3h −1.49%/−5.32%) to ~break-even
  (+0.5–0.75%). It cuts losses, but there is no underlying edge to harvest.

By an honest bar (Sharpe > 2, win > 50%, return > 2%), **only SBER qualifies.**

### Why — exogenous drivers, consistent with the SELL diagnosis

SBER is the most liquid, most momentum-driven MOEX name; its short-term moves are continuation-like
and readable from candle history. GAZP (gas) and LKOH (oil) are **commodity/news-driven** — their
moves are dominated by exogenous shocks (oil/gas prices, FX, geopolitics) that are absent from
OHLCV. This is the same mechanism the SELL diagnosis found: jump-like, exogenously-driven moves are
unpredictable from candle history. SBER's edge is special because its moves are comparatively
endogenous/technical.

## Implications for the architecture

- **The contract still works per ticker** — all three produce valid `ml_prediction`. Downstream
  risk_manager's "min expected edge" filter would naturally suppress GAZP/LKOH (low edge). The
  system design holds; it just wouldn't trade GAZP/LKOH much.
- **A 3-ticker portfolio on this rule would be SBER-dominated** — GAZP/LKOH add noise, not tradeable
  diversification. The "portfolio diversification" thesis does NOT hold with OHLCV-only models.
- **SBER is the only production-grade ticker so far.** GAZP/LKOH can be served (completeness, LLM
  fusion) but flagged as weak-edge / `is_production=false`. Packaging their artifacts is not
  worthwhile yet — they add no tradeable value.

## Recommended next steps

This result sharpens the roadmap into a clear fork:

1. **Orthogonal data (now specifically motivated).** GAZP needs gas/Brent; LKOH needs oil/USD-RUB;
   all benefit from IMOEX context. The exogenous drivers that make GAZP/LKOH unpredictable from
   OHLCV are exactly what these features supply. This is the path to (a) unlock GAZP/LKOH and
   (b) potentially lift SBER through the 0.48 ceiling. Bigger effort (data acquisition).

2. **Accept SBER-only and ship it.** Proceed to the final locked SBER test-set gate, package SBER as
   the single production ticker. GAZP/LKOH served as predictions but filtered downstream by edge.
   Fastest path to a usable product; revisit GAZP/LKOH later via orthogonal data.

Meta-labeling would help *filter* GAZP/LKOH noise but cannot create edge that isn't there — lower
priority than orthogonal data for these tickers.
