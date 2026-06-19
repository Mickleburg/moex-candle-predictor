# SBER H1 — Time Feature Ablation — 2026-06-02

## Hypothesis
hour_cos/hour_sin account for 58% of ExtraTrees feature importance in the frozen
candidate. Removing them may force the model to learn price structure and improve
generalisation, or may reveal that intraday seasonality is genuine signal.

## Method
- Model: ExtraTreesClassifier (n=300, depth=None, leaf=20, max_features=sqrt)
- Target: triple_barrier:h3:w12:up1.25:down1.25
- Walk-forward: 4 expanding folds, initial_train=12000, val_size=2000
- Seeds: [7, 42, 100]
- Removed features (no_time): hour_sin, hour_cos, dow_sin, dow_cos

## Results

| Metric | Baseline (27 feat) | No time (23 feat) | Delta |
|--------|-------------------|-------------------|-------|
| Val macro-F1 (mean ± std) | 0.4738 ± 0.0217 | 0.4097 ± 0.0088 | -0.0641 |
| Worst fold F1 | 0.4377 | 0.3965 | -0.0412 |
| SELL F1 | 0.4204 | 0.3235 | -0.0969 |
| HOLD F1 | 0.5815 | 0.5418 | -0.0398 |
| BUY F1 | 0.4195 | 0.3637 | -0.0557 |
| Time feature importance | 0.6219 | 0.0000 | — |

## Top-15 Feature Importances

### Baseline (with time features)

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | hour_cos ★ | 0.3247 |
| 2 | hour_sin ★ | 0.2441 |
| 3 | vol_8 | 0.0365 |
| 4 | dow_sin ★ | 0.0291 |
| 5 | vol_16 | 0.0277 |
| 6 | volume_ratio_20 | 0.0276 |
| 7 | volume_z_20 | 0.0244 |
| 8 | dow_cos ★ | 0.0240 |
| 9 | close_position_in_candle | 0.0205 |
| 10 | range_mean_8 | 0.0176 |
| 11 | vol_32 | 0.0165 |
| 12 | range_mean_32 | 0.0164 |
| 13 | range_mean_16 | 0.0154 |
| 14 | range_to_open | 0.0150 |
| 15 | ret_6 | 0.0146 |

### No-time (without hour/dow features)

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | volume_ratio_20 | 0.0975 |
| 2 | volume_z_20 | 0.0898 |
| 3 | vol_8 | 0.0698 |
| 4 | large_time_gap_flag | 0.0647 |
| 5 | range_mean_8 | 0.0567 |
| 6 | vol_16 | 0.0554 |
| 7 | close_position_in_candle | 0.0481 |
| 8 | range_to_open | 0.0479 |
| 9 | vol_32 | 0.0379 |
| 10 | range_mean_32 | 0.0374 |
| 11 | lower_shadow | 0.0352 |
| 12 | range_mean_16 | 0.0352 |
| 13 | body_abs | 0.0325 |
| 14 | ema_distance_16 | 0.0325 |
| 15 | ret_12 | 0.0324 |

## Note on Baseline Discrepancy

The walk-forward baseline (0.4738) is lower than the frozen candidate simple-split baseline (0.5792) because:
- Walk-forward folds 1–3 train on only 12–16k rows vs 17.5k for the simple split
- Folds 1–2 validate on 2024 data (earlier, potentially higher-noise period)
- The 0.5792 number reflects maximum training data (17.5k) evaluated on the final 15%

Both baselines confirm the same time-feature dominance: 62% of importance.
The relative delta Δ=−0.0641 is the reliable result here.

## Conclusion

**Time features (hour_sin/cos, dow_sin/cos) are genuine MOEX signal — not overfitting artifacts.**

Evidence:
- Removing them drops macro-F1 by −0.064 (−14% relative)
- SELL F1 drops most sharply (−0.097): intraday open/close patterns predict directional moves
- Without time features, volume and volatility take over (volume_ratio_20, vol_8) but insufficiently compensate
- `large_time_gap_flag` jumps from ~rank 27 to rank 4 — weekend/holiday gap pricing is important context
- No-time model variance drops (std 0.0088 vs 0.0217): less overfitting, but too much signal removed

**Decision: keep time features.** Intraday seasonality is a structural property of MOEX (fixed opening 10:00–18:45,
lunch break historically, Monday gap effects). ExtraTrees correctly leverages this.

The problem is not that time features exist — it is that their 62% dominance leaves little importance budget
for price features. Mitigation options:
1. **Add more price-structure features** (so price features compete): Word2Vec embeddings (Step 3)
2. **Constrain tree depth** to prevent time features from monopolising top splits
3. **Explicit time conditioning**: train separate models for morning/afternoon sessions

Next step: Step 2 — probability calibration.