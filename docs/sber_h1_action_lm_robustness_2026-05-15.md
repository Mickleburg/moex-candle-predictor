# SBER H1: robustness, calibration и BUY/SELL error analysis для LM-derived action features

Дата: 2026-05-15  
Статус: research-only. Production `/predict`, artifact format, final test evaluation и trading backtest не менялись.

## 1. Цель

Предыдущий этап показал, что `shape/gmm_diag/16 + lm_only` дает лучший downstream action signal среди небольшой сетки. Цель этого этапа - проверить, является ли этот эффект устойчивым, или это результат конкретного `random_state`/fold. Дополнительно проверяем, почему confidence/abstention не улучшает macro-F1, и где модель ошибается по `BUY`/`SELL`.

## 2. Почему проверяем robustness после LM features

`GMM`-словарь может зависеть от инициализации. Если downstream результат держится только на одном seed, его нельзя использовать как основу для следующего research-шага. Поэтому добавлен sweep по random states:

```text
7, 13, 21, 42, 100
```

Каждый seed заново fit-ит vocabulary внутри каждого train fold. Folds при этом не меняются.

## 3. Проверенные configs

Проверены ограниченные configs:

| vocabulary | feature set | classifier |
|---|---|---|
| `shape/gmm_diag/16` | `lm_only`, `base_lm_proba` | `logreg`, `ridge` |
| `shape/gmm_diag/20` | `lm_only`, `base_lm_proba` | `logreg`, `ridge` |
| `shape/kmeans/20` | `lm_only`, `base_lm_proba` | `logreg`, `ridge` |

`lm_only` означает scalar LM features + full next-word probability vector. `base_lm_proba` означает sentence/vectorizer features + LM probability features.

## 4. Leakage guarantees

Подтверждено кодом:

- train/validation folds ordered and non-overlapping;
- clusterer fit only на train fold;
- validation words assigned через train-fitted clusterer;
- word LM fit only на train fold words;
- vectorizer fit only на train samples;
- action classifier fit only на train samples;
- LM features используют только `words[t-L+1..t]`;
- actual future candle words не используются как features;
- action label использует future return только как supervised target;
- calibration, regime analysis и threshold sweep считаются только на validation folds;
- test split не используется.

## 5. Random-state stability

### Expanding

Лучший config по validation:

| vocabulary | feature set | classifier | macro-F1 mean | macro-F1 std | seed std | worst seed | worst fold | BUY F1 | SELL F1 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `shape/gmm_diag/20` | `lm_only` | `ridge` | 0.3988 | 0.0153 | 0.0033 | 0.3931 | 0.3810 | 0.2428 | 0.3434 |
| `shape/gmm_diag/20` | `lm_only` | `logreg` | 0.3979 | 0.0154 | 0.0034 | 0.3929 | 0.3800 | 0.2391 | 0.3440 |
| `shape/gmm_diag/16` | `lm_only` | `ridge` | 0.3972 | 0.0198 | 0.0027 | 0.3936 | 0.3731 | 0.2271 | 0.3545 |
| `shape/gmm_diag/16` | `lm_only` | `logreg` | 0.3960 | 0.0199 | 0.0036 | 0.3896 | 0.3693 | 0.2239 | 0.3540 |
| `shape/kmeans/20` | `lm_only` | `ridge` | 0.3786 | 0.0384 | 0.0043 | 0.3717 | 0.3402 | 0.2664 | 0.2768 |

### Rolling

Лучший config по validation:

| vocabulary | feature set | classifier | macro-F1 mean | macro-F1 std | seed std | worst seed | worst fold | BUY F1 | SELL F1 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `shape/gmm_diag/20` | `lm_only` | `ridge` | 0.4059 | 0.0137 | 0.0023 | 0.4020 | 0.3822 | 0.2535 | 0.3365 |
| `shape/gmm_diag/20` | `lm_only` | `logreg` | 0.4050 | 0.0143 | 0.0033 | 0.4010 | 0.3800 | 0.2509 | 0.3363 |
| `shape/gmm_diag/16` | `lm_only` | `ridge` | 0.4038 | 0.0215 | 0.0025 | 0.4000 | 0.3731 | 0.2387 | 0.3434 |
| `shape/gmm_diag/16` | `lm_only` | `logreg` | 0.4033 | 0.0231 | 0.0036 | 0.3986 | 0.3693 | 0.2358 | 0.3456 |
| `shape/kmeans/20` | `lm_only` | `ridge` | 0.3813 | 0.0410 | 0.0068 | 0.3689 | 0.3402 | 0.2577 | 0.2699 |

Интерпретация:

- `shape/gmm_diag/16 + lm_only` устойчив: результат не был удачным seed.
- Но `shape/gmm_diag/20 + lm_only` стабильно чуть лучше в expanding и rolling.
- Variance по random_state мала: примерно `0.002-0.003` macro-F1.
- Variance по folds выше: режимы рынка важнее инициализации.
- `shape/kmeans/20` заметно слабее downstream, хотя раньше был выбран как более интерпретируемый vocabulary для LM.

## 6. Word distribution stability

L1 до среднего распределения по fold x seed:

| vocabulary | режим | train L1 mean | val L1 mean |
|---|---|---:|---:|
| `shape/gmm_diag/16` | expanding | 0.8345 | 1.0229 |
| `shape/gmm_diag/20` | expanding | 0.7210 | 0.9171 |
| `shape/kmeans/20` | expanding | 1.1574 | 1.3293 |
| `shape/gmm_diag/16` | rolling | 0.7737 | 0.9692 |
| `shape/gmm_diag/20` | rolling | 0.7651 | 0.9789 |
| `shape/kmeans/20` | rolling | 1.1159 | 1.2869 |

Это не чистая seed-stability, а fold x seed distribution stability. Тем не менее `kmeans/20` явно более нестабилен по распределениям слов, а GMM-словари ближе друг к другу.

## 7. Calibration sanity-check

Calibration считалась только для моделей с `predict_proba`, то есть для `logreg`. Для `ridge` probabilities не выдумывались; decision scores использовались только как confidence-like score в старых threshold summaries.

Для `shape/gmm_diag/20 + lm_only + logreg`:

| режим | ECE | Brier | mean confidence | accuracy |
|---|---:|---:|---:|---:|
| expanding | 0.0351 | 0.6200 | 0.4608 | 0.4567 |
| rolling | 0.0366 | 0.6133 | 0.4662 | 0.4728 |

Reliability buckets показывают проблему не только в численной calibration, а в class composition:

| bucket | rolling accuracy | rolling confidence | rolling macro-F1 | pred HOLD share |
|---|---:|---:|---:|---:|
| `[0.33,0.40)` | 0.3634 | 0.3713 | 0.3554 | 0.3650 |
| `[0.40,0.50)` | 0.4108 | 0.4408 | 0.3797 | 0.4807 |
| `[0.50,0.60)` | 0.5672 | 0.5429 | 0.3523 | 0.8387 |
| `[0.60,0.70)` | 0.7166 | 0.6256 | 0.3100 | 0.9208 |

Вывод: high-confidence buckets могут иметь высокую accuracy, но macro-F1 падает, потому что модель становится HOLD-heavy. Это объясняет, почему naive abstention по confidence не работает для цели `BUY/SELL`.

## 8. LM confidence и uncertainty

По `lm_uncertainty` regime для `shape/gmm_diag/20 + lm_only + ridge`:

| regime | rolling macro-F1 | BUY F1 | SELL F1 | pred HOLD share |
|---|---:|---:|---:|---:|
| `low_entropy` | 0.2879 | 0.0408 | 0.0761 | 0.9617 |
| `mid_entropy` | 0.3640 | 0.2729 | 0.3519 | 0.4363 |
| `high_entropy` | 0.3059 | 0.3550 | 0.4362 | 0.0793 |

Низкая entropy LM не означает лучший action signal. Часто это означает, что LM уверенно ожидает доминирующее продолжение, а action classifier уходит в HOLD. Для BUY/SELL полезнее выглядят mid/high entropy зоны, но там растет неопределенность и нельзя механически делать trading filter.

## 9. Per-regime BUY/SELL error analysis

### Волатильность

Rolling:

| regime | macro-F1 | BUY F1 | SELL F1 | pred HOLD share |
|---|---:|---:|---:|---:|
| `low_vol` | 0.3796 | 0.1702 | 0.2729 | 0.7429 |
| `mid_vol` | 0.3902 | 0.2980 | 0.3715 | 0.4535 |
| `high_vol` | 0.3900 | 0.3507 | 0.4332 | 0.2175 |

BUY/SELL лучше ловятся в `mid_vol/high_vol`. В `low_vol` модель часто уходит в HOLD.

### Тренд

Rolling:

| regime | macro-F1 | BUY F1 | SELL F1 | pred HOLD share |
|---|---:|---:|---:|---:|
| `downtrend` | 0.4061 | 0.3090 | 0.3714 | 0.4555 |
| `flat` | 0.3847 | 0.1891 | 0.2835 | 0.6961 |
| `uptrend` | 0.4104 | 0.2625 | 0.3652 | 0.5409 |

Flat regimes хуже для action classes; directional regimes дают более полезный BUY/SELL signal.

### Сессия

Rolling:

| regime | macro-F1 | BUY F1 | SELL F1 | pred HOLD share |
|---|---:|---:|---:|---:|
| `early` | 0.3562 | 0.2579 | 0.4187 | 0.2832 |
| `middle` | 0.3851 | 0.2802 | 0.3047 | 0.5799 |
| `late` | 0.3946 | 0.2160 | 0.2512 | 0.7511 |

Early session лучше для SELL F1, middle лучше для BUY F1, late дает больше HOLD-like поведения.

## 10. Threshold sweep для BUY/SELL

Threshold sweep считался только для `logreg`, потому что нужны настоящие probabilities.

Схема:

```text
predict BUY only if p(BUY) >= buy_threshold
predict SELL only if p(SELL) >= sell_threshold
else HOLD
```

Validation-only best по macro-F1 для `shape/gmm_diag/20 + lm_only + logreg`:

| режим | avg best macro-F1 | BUY F1 | SELL F1 | action rate | частые thresholds |
|---|---:|---:|---:|---:|---|
| expanding | 0.4195 | 0.3156 | 0.3613 | 0.5587 | чаще `(0.30, 0.30)` или `(0.30, 0.35)` |
| rolling | 0.4219 | 0.3117 | 0.3565 | 0.5411 | чаще `(0.30, 0.30)` или `(0.30, 0.35)` |

Интерпретация: argmax вероятностей слишком консервативен для BUY/SELL. Низкие BUY/SELL thresholds повышают macro-F1 на validation, но это не production threshold и не trading rule. Следующий шаг - calibration/threshold selection внутри nested validation или walk-forward-only protocol.

## 11. Что подтвердилось

- Downstream signal от LM-derived features устойчив по random_state.
- `shape/gmm_diag/20 + lm_only` чуть лучше и стабильнее, чем `shape/gmm_diag/16 + lm_only`, в этой robustness-сетке.
- `shape/kmeans/20` слабее для downstream action, хотя остается интерпретируемым LM vocabulary.
- `base_lm_proba` не превосходит `lm_only`.
- Главная нестабильность идет по fold/regime, а не по seed.
- BUY/SELL лучше ловятся в directional и более волатильных regimes.

## 12. Что не подтвердилось

- Не подтвердилось, что high-confidence subset автоматически лучше.
- Не подтвердилось, что low LM entropy полезна для action decisions.
- Не подтвердилось, что добавление sentence features к LM probabilities улучшает лучший config.
- Не подтверждена торговая пригодность; backtest не делался.

## 13. Ограничения

- Test split не использовался.
- Threshold sweep validation-only и может переобучиться на validation, если использовать его как production selection.
- Regime thresholds fit-ятся на train fold, но сами regimes простые и диагностические.
- Ridge не имеет настоящих probabilities; calibration metrics считаются только для logreg.
- GMM проверен только с `diag` covariance.
- Нет повторов на других тикерах/timeframes.

## 14. Рекомендация

Для следующего research шага оставить:

```text
primary downstream config: shape/gmm_diag/20 + lm_only
control config: shape/gmm_diag/16 + lm_only
classifier: logreg для calibration/threshold experiments, ridge как score-only baseline
```

Переходить к GRU/TCN пока рано. Сначала лучше:

1. Сделать nested validation для BUY/SELL thresholds.
2. Добавить calibration метод (`CalibratedClassifierCV` или walk-forward calibration holdout) без test leakage.
3. Проверить class imbalance и thresholding на per-regime slices.
4. Сравнить `gmm_diag/20` на нескольких action horizons.
5. Только после этого рассматривать sequence model сложнее n-gram.
