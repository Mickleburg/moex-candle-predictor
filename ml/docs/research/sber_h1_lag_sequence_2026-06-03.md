# SBER H1 -- Lag Sequence Features -- 2026-06-03

## Hypothesis
Individual 1h lag returns expose price trajectory shape lost by cumulative returns.
Patterns like '3 bearish hours + volume spike' become learnable by ExtraTrees.

## Method
- Model: ExtraTreesClassifier (n=300, depth=None, leaf=20, sqrt)
- Target: triple_barrier:h3:w12:up1.25:down1.25
- Walk-forward: 4 folds, initial_train=12000, val=2000
- Seeds: [7, 42, 100]

## New lag features (26 total)
- lag_ret_2..lag_ret_10: individual 1h returns 2..10 steps back (9 features)
- lag_body_1..5: signed candle body k steps back (5 features)
- lag_vol_ratio_1..5: volume ratio k steps back (5 features)
- ret_day_1..3: ~1/2/3 trading-day cumulative returns (3 features)
- day_range, close_in_day_range: position within last-session high-low (2 features)
- up_streak, down_streak: consecutive up/down hourly returns (2 features)

## Results

| Condition | Features | F1 mean+-std | Worst fold | SELL | HOLD | BUY | Delta |
|-----------|---------|-------------|------------|------|------|-----|-------|
| baseline | 27 | 0.4738+-0.0217 | 0.4377 | 0.4204 | 0.5815 | 0.4195 | -- |
| lag_only | 26 | 0.3714+-0.0308 | 0.3367 | 0.2869 | 0.5337 | 0.2935 | -0.1024 |
| combined | 53 | 0.4711+-0.0219 | 0.4354 | 0.4251 | 0.5814 | 0.4067 | -0.0027 |

## Top-15 Feature Importances (combined)

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | hour_cos | 0.2602 |
| 2 | hour_sin | 0.1970 |
| 3 | close_in_day_range [LAG] | 0.0287 |
| 4 | dow_sin | 0.0236 |
| 5 | volume_ratio_20 | 0.0217 |
| 6 | dow_cos | 0.0215 |
| 7 | vol_8 | 0.0209 |
| 8 | lag_vol_ratio_3 [LAG] | 0.0207 |
| 9 | volume_z_20 | 0.0197 |
| 10 | lag_vol_ratio_5 [LAG] | 0.0188 |
| 11 | vol_16 | 0.0167 |
| 12 | lag_vol_ratio_2 [LAG] | 0.0161 |
| 13 | close_position_in_candle | 0.0159 |
| 14 | lag_vol_ratio_4 [LAG] | 0.0137 |
| 15 | down_streak [LAG] | 0.0124 |

## Conclusion

Lag features did not improve F1 (delta=-0.0027).
ExtraTrees cannot leverage temporal patterns even when lags are explicit.
This confirms LSTM is necessary: must move to Step 4 (sequence model).