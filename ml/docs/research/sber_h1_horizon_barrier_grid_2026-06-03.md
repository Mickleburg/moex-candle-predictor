# SBER H1 -- Horizon x Barrier Grid -- 2026-06-03

## Problem
At h>=6 with k=1.25, HOLD class disappears (price always hits barrier).
This paper shows barriers must scale with sqrt(horizon) to keep balanced labels.

## Grid (h x k), vol_window=12

| h | k | SELL% | HOLD% | BUY% | F1 mean+-std | Worst fold | SELL | HOLD | BUY | Delta |
|---|---|-------|-------|------|-------------|------------|------|------|-----|-------|
| 3 | 1.25 | 33% | 35% | 33% | 0.4738+-0.0217 | 0.4377 | 0.4204 | 0.5815 | 0.4195 | -0.0000 **BEST** |
| 3 | 1.75 | 26% | 49% | 26% | 0.3966+-0.0315 | 0.3558 | 0.2715 | 0.6770 | 0.2413 | -0.0772 |
| 6 | 1.75 | 36% | 26% | 38% | 0.4196+-0.0184 | 0.4043 | 0.4155 | 0.3619 | 0.4814 | -0.0542 |
| 6 | 2.0 | 34% | 32% | 35% | 0.4413+-0.0209 | 0.4134 | 0.3997 | 0.4816 | 0.4425 | -0.0325 |
| 6 | 2.5 | 29% | 43% | 29% | 0.4167+-0.0389 | 0.3711 | 0.2783 | 0.6319 | 0.3400 | -0.0571 |
| 12 | 2.5 | 39% | 20% | 41% | 0.3108+-0.0143 | 0.2894 | 0.4045 | 0.0150 | 0.5128 | -0.1630 |
| 12 | 3.0 | 35% | 29% | 36% | 0.3519+-0.0349 | 0.3019 | 0.3981 | 0.2005 | 0.4570 | -0.1219 |
| 12 | 3.5 | 31% | 38% | 31% | 0.4047+-0.0126 | 0.3858 | 0.3175 | 0.5290 | 0.3676 | -0.0691 |

## Conclusion

Best config h=3,k=1.25 gives macro-F1=0.4738 (delta=-0.0000).

No meaningful improvement from changing horizon.
The signal ceiling at this resolution is ~0.47-0.49.

**Root cause**: 1H MOEX triple-barrier prediction is inherently hard at any horizon.
Intraday price moves are largely random noise; the 62% time-feature importance
shows the model mostly predicts WHEN to trade (session open/close effects),
not WHERE price will go.

Next steps with higher expected impact:
1. Transformer with attention over 32-step sequences (captures long-range deps)
2. Additional MOEX-specific features: macro (CBR rate, USD/RUB), sector flow
3. Multi-ticker joint training (SBER+LKOH+GAZP → 3x data, shared patterns)
4. Pre-train on unsupervised next-candle prediction, then fine-tune on labels