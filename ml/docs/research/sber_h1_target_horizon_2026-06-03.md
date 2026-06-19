# SBER H1 -- Target Horizon Ablation -- 2026-06-03

## Hypothesis
Triple-barrier h=3 is too noisy. Longer horizon = cleaner labels = higher F1.

- Model: ExtraTreesClassifier (frozen candidate spec)
- vol_window=12, up_k=down_k=1.25
- Walk-forward: 4 folds, initial_train=12000, val=2000
- Seeds: [7, 42, 100]

## Results

| Horizon | F1 mean+-std | Worst fold | SELL | HOLD | BUY | Delta vs h=3 | Valid labels |
|---------|-------------|------------|------|------|-----|-------------|-------------|
| h=3 | 0.4738+-0.0217 | 0.4377 | 0.4204 | 0.5815 | 0.4195 | +0.0000 | 21301 |
| h=4 | 0.4600+-0.0182 | 0.4289 | 0.4192 | 0.4938 | 0.4669 | -0.0138 | 21301 <-- |
| h=6 | 0.3137+-0.0100 | 0.2982 | 0.4081 | 0.0000 | 0.5329 | -0.1601 | 21301 |
| h=8 | 0.3219+-0.0110 | 0.3050 | 0.4094 | 0.0000 | 0.5564 | -0.1519 | 21301 |
| h=12 | 0.3245+-0.0084 | 0.3116 | 0.4044 | 0.0000 | 0.5692 | -0.1493 | 21301 |

## Class Distribution by Horizon

| Horizon | SELL% | HOLD% | BUY% |
|---------|-------|-------|------|
| h=3 | 32.6% | 34.6% | 32.8% |
| h=4 | 36.1% | 26.8% | 37.1% |
| h=6 | 40.4% | 16.9% | 42.7% |
| h=8 | 42.5% | 12.1% | 45.4% |
| h=12 | 43.9% | 9.0% | 47.1% |

## Conclusion

Horizon changes give marginal gains (best delta=+0.0000).
The performance ceiling is not driven by label noise at h=3.
The signal itself is limited at this timeframe+ticker combination.
Next: try Transformer architecture or multi-ticker training.