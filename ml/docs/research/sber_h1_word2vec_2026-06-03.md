# SBER H1 — Word2Vec Candle Embeddings — 2026-06-03

## Hypothesis
Co-occurrence SVD embeddings of candle shape clusters add sequential price-structure
context that complements the flat continuous_regime feature snapshot.
Combining both should improve macro-F1 above baseline 0.4675 (walk-forward CV).

## Method
- Model: ExtraTreesClassifier (n=300, depth=None, leaf=20, sqrt)
- Target: triple_barrier:h3:w12:up1.25:down1.25
- Walk-forward: 4 folds, initial_train=12000, val=2000
- Seeds: [7, 42, 100]
- W2V pipeline: normalize_ohlc → KMeans(nw) → co-occurrence SVD(nv) → context mean(nm)
- Grid: nw∈[20, 30, 50], nv∈[16, 32], nm∈[10, 20]

## Results — Best Configs

| Condition | Features | macro-F1 (mean±std) | Worst fold | SELL | HOLD | BUY | Δ vs baseline |
|-----------|---------|---------------------|------------|------|------|-----|--------------|
| baseline | 27 | 0.4738 ± 0.0217 | 0.4377 | 0.4204 | 0.5815 | 0.4195 | — |
| w2v_only (best) | 32 | 0.3544 ± 0.0271 | 0.3341 | 0.2911 | 0.4414 | 0.3308 | -0.1194 |
| w2v_combined (best) | 43 | 0.4717 ± 0.0207 | 0.4371 | 0.4077 | 0.5829 | 0.4245 | -0.0021 |

  Best w2v_only config: nw30_nv32_nm10
  Best w2v_combined config: nw30_nv16_nm10

## Grid Search Results

| Config | w2v_only F1 | w2v_combined F1 | Δ combined |
|--------|------------|-----------------|-----------|
| nw30_nv16_nm10 | 0.3449 | 0.4717 | -0.0021 |
| nw50_nv16_nm10 | 0.3517 | 0.4684 | -0.0054 |
| nw20_nv16_nm20 | 0.3450 | 0.4665 | -0.0073 |
| nw50_nv32_nm10 | 0.3518 | 0.4664 | -0.0074 |
| nw30_nv16_nm20 | 0.3179 | 0.4661 | -0.0077 |
| nw50_nv16_nm20 | 0.3236 | 0.4661 | -0.0077 |
| nw30_nv32_nm10 | 0.3544 | 0.4648 | -0.0090 |
| nw20_nv32_nm20 | 0.3142 | 0.4641 | -0.0097 |
| nw50_nv32_nm20 | 0.3296 | 0.4639 | -0.0098 |
| nw30_nv32_nm20 | 0.3209 | 0.4624 | -0.0114 |
| nw20_nv16_nm10 | 0.3497 | 0.4608 | -0.0130 |
| nw20_nv32_nm10 | 0.3319 | 0.4560 | -0.0177 |

## Top-10 Feature Importances — Best w2v_combined

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | hour_cos | 0.2628 |
| 2 | hour_sin | 0.2017 |
| 3 | dow_sin | 0.0255 |
| 4 | w2v_2 | 0.0244 |
| 5 | vol_8 | 0.0228 |
| 6 | volume_ratio_20 | 0.0226 |
| 7 | dow_cos | 0.0226 |
| 8 | w2v_9 | 0.0205 |
| 9 | volume_z_20 | 0.0183 |
| 10 | vol_16 | 0.0175 |

## Conclusion

W2V embeddings did not improve over baseline (Δ=-0.0021).
SVD co-occurrence does not capture the same structure as neural skip-gram.
Recommendation: install gensim on a Python 3.11/3.12 environment and re-run,
OR move to Step 4 (LSTM) which directly models sequence structure.