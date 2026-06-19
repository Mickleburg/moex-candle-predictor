# SBER H1: nested calibration и BUY/SELL threshold selection

Дата: 2026-05-15  
Статус: research-only. Production `/predict`, artifact bundle, final test evaluation и trading backtest не менялись.

## 1. Цель

Предыдущий validation-only threshold sweep улучшал macro-F1, но мог переобучаться к validation fold. Цель этого этапа - проверить, сохраняется ли прирост в честной nested схеме:

```text
outer train range -> inner_train + calibration
outer validation range -> untouched evaluation
```

Thresholds, temperature и regime-specific правила выбираются только на `calibration`. Outer validation используется только для оценки.

## 2. Почему нужен nested protocol

Обычный threshold sweep по validation отвечает на вопрос “можно ли подобрать пороги под эти данные”. Это полезно как upper-bound диагностика, но не как честная оценка. Nested protocol отделяет:

- обучение representation/model на `inner_train`;
- выбор calibration/thresholds на `calibration`;
- итоговую оценку на untouched `outer_validation`.

Test split не используется.

## 3. Схема split

Использован rolling режим:

```text
train_size = 12000
calibration_size = 2500
outer validation size = 3000
step_size = 3000
max_folds = 4
```

Для каждого outer fold:

```text
inner_train = first 9500 rows of outer train
calibration = last 2500 rows of outer train
outer_validation = next 3000 rows
```

Sentence windows и action labels не пересекают границы `inner_train`, `calibration` или `outer_validation`.

## 4. Проверенные configs

| параметр | значения |
|---|---|
| vocabulary | `shape/gmm_diag/20`, `shape/gmm_diag/16` |
| feature set | `lm_only` |
| classifier | `logreg` |
| class weight | `none`, `balanced` |
| action horizon | `1`, `3` |
| LM context | `16` |
| LM forecast horizon | `3` |
| LM order / alpha | `2 / 0.1` |

`lm_only` = scalar LM features + full next-word probability vector. Фактические future candle words не используются как признаки.

## 5. Calibration methods

Проверена простая temperature scaling:

```text
p_calibrated = softmax(log(p + eps) / T)
T in [0.75, 1.0, 1.25, 1.5, 2.0]
```

`T` выбирается на calibration split вместе с thresholds по `macro_f1`.

## 6. BUY/SELL threshold modes

Проверены режимы:

| mode | описание |
|---|---|
| `argmax` | baseline: `argmax p(SELL/HOLD/BUY)` |
| `global` | единые BUY/SELL thresholds, выбранные на calibration |
| `regime_volatility` | thresholds отдельно для low/mid/high volatility, fallback на global |
| `regime_trend` | thresholds отдельно для down/flat/up trend, fallback на global |
| `oracle_global` | diagnostic upper bound: thresholds выбраны на outer validation, leakage/oracle |

`oracle_global` явно помечен как `is_oracle=true` и не участвует в выборе best honest config.

## 7. Class weighting

Проверены:

- `class_weight=None`;
- `class_weight=balanced`.

Цель была не поднять accuracy, а проверить macro-F1 и BUY/SELL F1 без полного разрушения HOLD.

## 8. Action horizon sensitivity

Проверены action horizons:

- `horizon=1`;
- `horizon=3`.

LM forecast horizon оставался `K=3`; action horizon относится только к построению `SELL/HOLD/BUY` label.

## 9. Результаты: argmax baseline

Лучшие argmax rows:

| vocabulary | horizon | class weight | macro-F1 | worst fold | BUY F1 | SELL F1 | action rate |
|---|---:|---|---:|---:|---:|---:|---:|
| `shape/gmm_diag/20` | 1 | none | 0.4037 | 0.3898 | 0.3188 | 0.2632 | 0.404 |
| `shape/gmm_diag/20` | 1 | balanced | 0.3984 | 0.3851 | 0.2535 | 0.3131 | 0.391 |
| `shape/gmm_diag/16` | 1 | balanced | 0.3949 | 0.3758 | 0.2244 | 0.3358 | 0.397 |
| `shape/gmm_diag/16` | 1 | none | 0.3930 | 0.3580 | 0.3163 | 0.2368 | 0.393 |

Для `horizon=3` argmax заметно хуже по macro-F1:

| vocabulary | horizon | class weight | macro-F1 | BUY F1 | SELL F1 | action rate |
|---|---:|---|---:|---:|---:|---:|
| `shape/gmm_diag/20` | 3 | balanced | 0.3187 | 0.1940 | 0.3127 | 0.342 |
| `shape/gmm_diag/16` | 3 | balanced | 0.3185 | 0.1829 | 0.3236 | 0.342 |

## 10. Calibration-selected thresholds

Лучший честный config:

```text
shape/gmm_diag/20 + lm_only + logreg
class_weight = balanced
action_horizon = 1
threshold_mode = global
```

Результат:

| metric | value |
|---|---:|
| outer validation macro-F1 mean | 0.4187 |
| std | 0.0059 |
| worst fold | 0.4111 |
| BUY F1 mean | 0.3031 |
| SELL F1 mean | 0.3526 |
| mean action rate | 0.5351 |

Выбранные thresholds по folds:

| fold | T | buy threshold | sell threshold | calibration macro-F1 | outer macro-F1 | action rate |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.00 | 0.30 | 0.30 | 0.4105 | 0.4202 | 0.4050 |
| 2 | 1.25 | 0.30 | 0.25 | 0.4298 | 0.4272 | 0.7187 |
| 3 | 1.00 | 0.35 | 0.30 | 0.4163 | 0.4111 | 0.5168 |
| 4 | 0.75 | 0.30 | 0.30 | 0.4314 | 0.4162 | 0.5000 |

Интерпретация: честный threshold selection сохраняет прирост относительно argmax. Он не является только артефактом validation-only sweep.

## 11. Oracle upper bound

Лучшие oracle rows:

| vocabulary | horizon | class weight | macro-F1 | worst fold | BUY F1 | SELL F1 | action rate |
|---|---:|---|---:|---:|---:|---:|---:|
| `shape/gmm_diag/20` | 1 | none | 0.4247 | 0.4226 | 0.3240 | 0.3740 | 0.618 |
| `shape/gmm_diag/20` | 1 | balanced | 0.4234 | 0.4173 | 0.3290 | 0.3554 | 0.601 |
| `shape/gmm_diag/16` | 1 | none | 0.4225 | 0.4038 | 0.3321 | 0.3503 | 0.582 |

Gap между honest best `0.4187` и oracle best `0.4247` небольшой, но он есть. Это означает, что calibration selection работает неплохо, но часть threshold потенциала все еще нестабильна.

## 12. Regime-conditioned thresholds

Regime thresholds не обогнали global best:

| config | mode | macro-F1 | BUY F1 | SELL F1 | action rate |
|---|---|---:|---:|---:|---:|
| `gmm20 h1 none` | `regime_trend` | 0.4185 | 0.3245 | 0.3252 | 0.516 |
| `gmm20 h1 none` | `regime_volatility` | 0.4176 | 0.3440 | 0.3186 | 0.549 |
| `gmm20 h1 balanced` | `global` | 0.4187 | 0.3031 | 0.3526 | 0.535 |

Вывод: regime-specific thresholds полезны как диагностика, но пока не дают стабильного выигрыша над global thresholds. Вероятная причина - мало calibration samples на отдельные regimes и нестабильность BUY/SELL trade-off.

## 13. Per-regime results для best honest config

Best honest config: `shape/gmm_diag/20 + lm_only + logreg + balanced + horizon=1 + global`.

### Volatility

| regime | macro-F1 | BUY F1 | SELL F1 | HOLD F1 | action rate |
|---|---:|---:|---:|---:|---:|
| `low_vol` | 0.4050 | 0.2365 | 0.3076 | 0.6709 | 0.3907 |
| `mid_vol` | 0.3998 | 0.3528 | 0.4142 | 0.4324 | 0.7238 |
| `high_vol` | 0.3670 | 0.4058 | 0.4163 | 0.2788 | 0.8894 |

Thresholding поднимает BUY/SELL в mid/high volatility, но HOLD качество падает при высокой action rate.

### Trend

| regime | macro-F1 | BUY F1 | SELL F1 | HOLD F1 | action rate |
|---|---:|---:|---:|---:|---:|
| `downtrend` | 0.4099 | 0.3543 | 0.3983 | 0.4771 | 0.6718 |
| `flat` | 0.4119 | 0.2560 | 0.3140 | 0.6658 | 0.4207 |
| `uptrend` | 0.4134 | 0.3088 | 0.3632 | 0.5681 | 0.5841 |

Directional regimes остаются полезными, но global thresholds уже достаточно хорошо балансируют action/HOLD.

### Session

| regime | macro-F1 | BUY F1 | SELL F1 | HOLD F1 | action rate |
|---|---:|---:|---:|---:|---:|
| `early` | 0.3298 | 0.2848 | 0.4334 | 0.2713 | 0.8300 |
| `middle` | 0.4017 | 0.3353 | 0.3249 | 0.5450 | 0.5521 |
| `late` | 0.4161 | 0.2725 | 0.2759 | 0.6999 | 0.3479 |

Early session дает много action predictions и высокий SELL F1, но низкий HOLD F1. Late session более HOLD-heavy.

### LM uncertainty

| regime | macro-F1 | BUY F1 | SELL F1 | HOLD F1 | action rate |
|---|---:|---:|---:|---:|---:|
| `low_entropy` | 0.3519 | 0.1573 | 0.1642 | 0.7342 | 0.1611 |
| `mid_entropy` | 0.3232 | 0.3323 | 0.3920 | 0.2453 | 0.7937 |
| `high_entropy` | 0.2611 | 0.3605 | 0.4228 | 0.0000 | 1.0000 |

Как и раньше, low LM entropy не означает хороший action signal. High entropy агрессивно уходит в action classes и теряет HOLD.

## 14. Class weights

Для `shape/gmm_diag/20`, `horizon=1`:

| mode | class weight | macro-F1 | BUY F1 | SELL F1 | action rate |
|---|---|---:|---:|---:|---:|
| `global` | balanced | 0.4187 | 0.3031 | 0.3526 | 0.535 |
| `global` | none | 0.4164 | 0.3331 | 0.3164 | 0.537 |
| `argmax` | none | 0.4037 | 0.3188 | 0.2632 | 0.404 |
| `argmax` | balanced | 0.3984 | 0.2535 | 0.3131 | 0.391 |

`balanced` лучше для честного global threshold по macro-F1 и SELL F1, но `none` дает выше BUY F1.

## 15. Action horizon comparison

Лучший честный `horizon=1`:

```text
shape/gmm_diag/20 + balanced + global
macro-F1 = 0.4187
BUY F1 = 0.3031
SELL F1 = 0.3526
```

Лучший честный `horizon=3`:

```text
shape/gmm_diag/16 + balanced + global
macro-F1 = 0.3707
BUY F1 = 0.3575
SELL F1 = 0.3649
```

`horizon=3` улучшает BUY/SELL F1, но сильно снижает общий macro-F1 и повышает action rate. Это не очевидное улучшение, а другая trade-off постановка.

## 16. Что реально улучшилось

- Honest global threshold selection улучшил `horizon=1` macro-F1 относительно argmax:

```text
argmax best: 0.4037
nested global best: 0.4187
oracle global best: 0.4247
```

- BUY/SELL стали более сбалансированными, особенно SELL F1.
- Прирост сохраняется на untouched outer validation.

## 17. Что не улучшилось

- Regime-conditioned thresholds не победили global thresholds.
- `horizon=3` не улучшил macro-F1, хотя BUY/SELL F1 выше.
- Calibration/thresholding не решают regime instability полностью.
- High-confidence/low-entropy логика все еще не является готовым фильтром.

## 18. Leakage guarantees

Подтверждено кодом и smoke checks:

- calibration split строго внутри outer train и раньше outer validation;
- classifier fit только на inner_train;
- calibration samples не входят в classifier fit;
- thresholds и temperature выбираются только на calibration;
- outer validation не используется для honest threshold selection;
- oracle rows явно помечены `is_oracle=true` и не участвуют в `best_honest`;
- thresholds выбираются из заданного grid;
- temperature выбирается из заданного grid;
- probabilities finite and sum to 1;
- action horizon не выходит за split boundary;
- regime thresholds fallback работает при малом числе calibration samples;
- test split не используется.

## 19. Ограничения

- Проверен только `logreg`, потому что нужны настоящие probabilities.
- Temperature scaling простая, без отдельной isotonic/Platt calibration.
- Threshold grid грубый.
- Regime-specific selection может требовать больше calibration данных.
- Нет final test evaluation.
- Нет trading backtest.
- Нет production thresholds.

## 20. Вывод

Честный nested protocol подтвердил, что BUY/SELL thresholding действительно улучшает downstream action quality для `horizon=1`:

```text
best honest config:
shape/gmm_diag/20 + lm_only + logreg + class_weight=balanced
action_horizon=1
global thresholds selected on calibration
outer validation macro-F1 = 0.4187
```

Это не полностью артефакт validation-only sweep. Однако oracle gap показывает, что thresholds остаются нестабильными по fold, а regime-specific пороги пока не дают надежного выигрыша.

Следующий шаг: улучшать не архитектуру модели, а calibration protocol:

1. Попробовать более аккуратный calibration holdout или rolling nested selection с несколькими calibration windows.
2. Сделать более тонкий threshold grid вокруг `0.25-0.35`.
3. Проверить class-specific objective, где BUY/SELL F1 учитываются явно.
4. Сохранять `shape/gmm_diag/20 + lm_only + logreg` как primary research config.
5. Не переходить к GRU/TCN, пока threshold/calibration baseline дает честный прирост.
