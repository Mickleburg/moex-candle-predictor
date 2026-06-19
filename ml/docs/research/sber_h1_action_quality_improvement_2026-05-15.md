# SBER H1: улучшение downstream action quality без нейросетей

Дата: 2026-05-15  
Статус: research-only. Production `/predict`, final test evaluation, artifact bundle и trading backtest не менялись.

## 1. Цель

Цель этапа - честно превысить текущий nested baseline:

```text
shape/gmm_diag/20 + lm_only + logreg
class_weight = balanced
action_horizon = 1
global thresholds selected on calibration
outer validation macro-F1 = 0.4187
```

Ограничения сохранены: без test leakage, без future candle words как features, без GRU/TCN/Transformer.

## 2. Baseline

Baseline прошлого этапа:

| config | macro-F1 | worst fold | BUY F1 | SELL F1 | action rate |
|---|---:|---:|---:|---:|---:|
| `gmm20 + lm_only + logreg + balanced + global thresholds` | 0.4187 | 0.4111 | 0.3031 | 0.3526 | 0.5351 |

Argmax baseline прошлого этапа был ниже:

```text
macro-F1 = 0.4037
```

## 3. Что проверяли

Проверены:

- past-only regime features как признаки модели;
- finer BUY/SELL threshold grid;
- более тонкая temperature grid;
- objectives `macro_f1`, `buy_sell_hmean_f1`, `macro_f1_action_penalty`;
- class weights `balanced`, `action_boost_1.2`;
- primary vocabulary `shape/gmm_diag/20`.

Полная сетка из задания была запущена, но превысила runtime limit. Поэтому выполнена сокращенная честная сетка:

```text
vocab: shape:gmm:20
feature sets: lm_only,lm_regime
class weights: balanced,action_boost_1.2
objectives: macro_f1,buy_sell_hmean_f1
threshold mode: global
action horizon: 1
```

Дополнительно отдельным запуском проверен `macro_f1_action_penalty` для `lm_regime`.

Ensemble не реализован в этом проходе: сначала нужно было проверить более дешевую гипотезу про regime features/objective/weights. Полная сетка уже уперлась в лимит, поэтому ensemble добавил бы слишком много степеней свободы.

## 4. Nested protocol

Для каждого rolling fold:

```text
outer train -> inner_train + calibration
outer validation -> untouched evaluation
```

Модель обучается только на `inner_train`. Thresholds и temperature выбираются только на `calibration`. Outer validation используется только для оценки.

## 5. Leakage guarantees

Подтверждено кодом:

- clusterer fit only on inner_train;
- LM fit only on inner_train words;
- classifier fit only on inner_train;
- calibration не входит в classifier fit;
- regime thresholds fit only on inner_train;
- calibration и outer validation только получают assigned regimes через train-fitted thresholds;
- action labels не выходят за split boundary;
- future candle words не используются как features;
- outer validation не используется для honest threshold selection;
- oracle rows помечены `is_oracle=true` и исключены из best honest selection;
- test split не используется.

## 6. Feature sets

| feature set | состав |
|---|---|
| `lm_only` | scalar LM features + full next-word probability vector |
| `lm_regime` | scalar LM features + one-hot regimes + continuous past-only regime features |
| `lm_regime_proba` | поддержан в CLI, но не вошел в сокращенный основной запуск из-за runtime |

Regime features:

- volatility bucket `low/mid/high`;
- trend bucket `down/flat/up`;
- session bucket `early/middle/late`;
- LM uncertainty bucket `low/mid/high entropy`;
- standardized past volatility;
- standardized past return/trend;
- standardized LM entropy;
- standardized LM top1 probability;
- standardized LM top3 mass.

Все пороги и стандартизация fit-ятся только на `inner_train`.

## 7. Лучший честный config

Новый лучший честный результат:

```text
shape/gmm_diag/20 + lm_regime + logreg
class_weight = action_boost_1.2
action_horizon = 1
threshold_mode = argmax
```

| metric | value |
|---|---:|
| macro-F1 mean | 0.4265 |
| macro-F1 std | 0.0034 |
| worst fold macro-F1 | 0.4223 |
| BUY F1 | 0.3618 |
| SELL F1 | 0.3306 |
| BUY/SELL hmean F1 | 0.3447 |
| action rate | 0.5973 |

Сравнение с baseline:

```text
old nested baseline: 0.4187
new best:            0.4265
absolute gain:       +0.0078
```

Worst fold также улучшился:

```text
old worst fold: 0.4111
new worst fold: 0.4223
```

## 8. Результаты по config

Лучшие honest rows:

| feature set | class weight | objective | mode | macro-F1 | worst | BUY F1 | SELL F1 | action rate |
|---|---|---|---|---:|---:|---:|---:|---:|
| `lm_regime` | `action_boost_1.2` | argmax | argmax | 0.4265 | 0.4223 | 0.3618 | 0.3306 | 0.597 |
| `lm_regime` | `action_boost_1.2` | macro_f1 | global | 0.4224 | 0.4122 | 0.3265 | 0.3551 | 0.579 |
| `lm_regime` | balanced | macro_f1 | global | 0.4212 | 0.4164 | 0.2985 | 0.3820 | 0.572 |
| `lm_only` | balanced | macro_f1 | global | 0.4158 | 0.4008 | 0.3031 | 0.3429 | 0.520 |
| `lm_only` | `action_boost_1.2` | macro_f1 | global | 0.4138 | 0.4037 | 0.3245 | 0.3163 | 0.533 |

Главный вывод: прирост дал не finer thresholding сам по себе, а добавление regime features и action-boosted class weights. Для нового feature set argmax оказался сильнее global thresholding.

## 9. Objectives

### `macro_f1`

Лучший threshold row:

```text
lm_regime + action_boost_1.2 + global thresholds
macro-F1 = 0.4224
```

Это выше старого baseline `0.4187`, но ниже нового argmax `0.4265`.

### `buy_sell_hmean_f1`

Этот objective улучшает баланс BUY/SELL, но слишком агрессивно поднимает action rate:

| config | macro-F1 | BUY F1 | SELL F1 | hmean | action rate |
|---|---:|---:|---:|---:|---:|
| `lm_regime + action_boost_1.2` | 0.3969 | 0.3551 | 0.3784 | 0.3649 | 0.777 |
| `lm_regime + balanced` | 0.3926 | 0.3285 | 0.4064 | 0.3624 | 0.797 |

Вывод: action-class objective полезен для диагностики BUY/SELL, но снижает общий macro-F1 из-за просадки HOLD.

### `macro_f1_action_penalty`

Отдельный запуск:

| config | macro-F1 | BUY F1 | SELL F1 | action rate |
|---|---:|---:|---:|---:|
| `lm_regime + action_boost_1.2 + global` | 0.4147 | 0.3102 | 0.3261 | 0.495 |
| `lm_regime + balanced + global` | 0.4124 | 0.2819 | 0.3476 | 0.476 |

Penalty успешно удерживает action rate около `0.50`, но macro-F1 падает. Пока не лучший objective.

## 10. Class weights

`action_boost_1.2` оказался полезен вместе с `lm_regime`:

| config | macro-F1 | BUY F1 | SELL F1 | action rate |
|---|---:|---:|---:|---:|
| `lm_regime + action_boost_1.2 + argmax` | 0.4265 | 0.3618 | 0.3306 | 0.597 |
| `lm_regime + balanced + argmax` | 0.4012 | 0.2199 | 0.3524 | 0.384 |

`action_boost_1.2` поднимает BUY сильно, SELL остается приемлемым, HOLD не разваливается полностью. Более сильный `action_boost_1.5` не был запущен в сокращенной сетке из-за runtime; его стоит проверять отдельно и осторожно.

## 11. Regime post-analysis для лучшего config

Best config: `gmm20 + lm_regime + action_boost_1.2 + argmax`.

### Volatility

| regime | macro-F1 | BUY F1 | SELL F1 | HOLD F1 | action rate |
|---|---:|---:|---:|---:|---:|
| low_vol | 0.4140 | 0.2699 | 0.2978 | 0.6744 | 0.403 |
| mid_vol | 0.3673 | 0.4205 | 0.3890 | 0.2923 | 0.860 |
| high_vol | 0.2901 | 0.4479 | 0.3186 | 0.1037 | 0.970 |

Улучшение BUY/SELL концентрируется в mid/high volatility, но там резко падает HOLD.

### Trend

| regime | macro-F1 | BUY F1 | SELL F1 | HOLD F1 | action rate |
|---|---:|---:|---:|---:|---:|
| downtrend | 0.3926 | 0.3920 | 0.3745 | 0.4112 | 0.782 |
| flat | 0.4225 | 0.2669 | 0.3293 | 0.6714 | 0.425 |
| uptrend | 0.4076 | 0.4234 | 0.2723 | 0.5271 | 0.687 |

Flat regime выигрывает за счет HOLD, directional regimes дают больше BUY/SELL.

### Session

| regime | macro-F1 | BUY F1 | SELL F1 | HOLD F1 | action rate |
|---|---:|---:|---:|---:|---:|
| early | 0.2824 | 0.1338 | 0.4878 | 0.2256 | 0.876 |
| middle | 0.3631 | 0.4485 | 0.1437 | 0.4972 | 0.688 |
| late | 0.3933 | 0.3562 | 0.1257 | 0.6981 | 0.357 |

Session effect стал очень асимметричным: early лучше для SELL, middle/late лучше для BUY/HOLD.

### LM uncertainty

| regime | macro-F1 | BUY F1 | SELL F1 | HOLD F1 | action rate |
|---|---:|---:|---:|---:|---:|
| low_entropy | 0.3703 | 0.2452 | 0.1495 | 0.7160 | 0.264 |
| mid_entropy | 0.3330 | 0.4183 | 0.3569 | 0.2237 | 0.863 |
| high_entropy | 0.3051 | 0.3957 | 0.4192 | 0.1004 | 0.961 |

Как и раньше, low entropy не является хорошим action фильтром. High/mid entropy дают BUY/SELL, но почти убивают HOLD.

## 12. Что улучшилось

- Macro-F1 честно вырос с `0.4187` до `0.4265`.
- Worst fold вырос с `0.4111` до `0.4223`.
- BUY F1 вырос с `0.3031` до `0.3618`.
- BUY/SELL hmean вырос с примерно `0.327` у старого baseline до `0.3447`.
- Улучшение пришло от past-only regime features + custom action class weight.

## 13. Что не улучшилось

- Fine thresholds не стали главным источником улучшения.
- `buy_sell_hmean_f1` objective слишком агрессивно увеличивает action rate.
- `macro_f1_action_penalty` держит action rate, но снижает macro-F1.
- HOLD остается уязвимым в mid/high volatility и high entropy regimes.
- Ensemble не проверен из-за runtime.

## 14. Ограничения

- Основной запуск был сокращен после timeout полной сетки.
- Не проверены `shape/gmm_diag/16`, `lm_regime_proba`, `action_boost_1.5` в полном rolling run.
- Не выполнен ensemble.
- Не использовался test.
- Нет trading backtest.
- Нет production thresholds.

## 15. Вывод

Качество удалось честно сдвинуть выше текущего baseline:

```text
baseline: 0.4187
new best: 0.4265
gain: +0.0078
```

Новый research-primary config для следующего шага:

```text
shape/gmm_diag/20 + lm_regime + logreg
class_weight = action_boost_1.2
action_horizon = 1
decision = argmax
```

Закреплять production пока рано. Следующий research шаг:

1. Проверить `lm_regime_proba` и `action_boost_1.5` в более узкой сетке.
2. Повторить лучший config на нескольких random_state.
3. Проверить, не является ли argmax improvement следствием class-weight over-action.
4. Сделать отдельный objective, который ограничивает action rate per-regime, а не глобально.
5. Только после этого возвращаться к threshold/calibration или ensemble.
