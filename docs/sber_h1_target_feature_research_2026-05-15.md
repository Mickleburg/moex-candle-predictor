# SBER H1: target redesign и continuous past-only features

## 1. Цель

Цель этапа - перестать выжимать последние сотые из уже исчерпанной ветки `gmm16/gmm20 + lm_regime + threshold/class_weight` и проверить новые рычаги качества:

- альтернативные action targets;
- continuous past-only признаки;
- более сильные baseline-модели без нейросетей.

Test split в этом этапе не используется.

## 2. Почему перестали крутить старую narrow-сетку

Пользователь локально добил полную validation-only narrow-сетку:

```text
vocab: shape/gmm:20, shape/gmm:16
features: lm_regime, lm_regime_proba
weights: balanced, action_boost_1.2, action_boost_1.5
modes: argmax, global
random_states: 7,13,21,42,100
```

Лучшим честным config остался:

```text
shape/gmm_diag/20 + lm_regime + action_boost_1.2 + argmax

macro-F1:    0.4238
worst:       0.4057
BUY F1:      0.3603
SELL F1:     0.3281
action_rate: 0.5983
```

`shape/gmm_diag/16` близок, но не лучше по mean. `lm_regime_proba` не дает надежного прироста. `global thresholds` хуже `argmax`, а `oracle_global` около `0.428-0.430`, то есть запас от thresholding небольшой.

Вывод: дальше нужен не очередной threshold grid, а изменение target/data/features.

## 3. Локальные результаты пользователя

Ключевые локальные validation-only выводы зафиксированы отдельно:

```text
docs/sber_h1_local_validation_runs_2026-05-15.md
```

Final untouched test уже был выполнен один раз для frozen candidate:

```text
test macro-F1 = 0.4055
BUY F1        = 0.2553
SELL F1       = 0.3113
HOLD F1       = 0.6499
```

Этот test не используется для новых сравнений и не должен использоваться для выбора новых candidates.

## 4. Новые target modes

Добавлен research-only модуль:

```text
ml/src/nlp/targets.py
```

Поддержаны target modes:

| mode | смысл |
|---|---|
| `return_threshold` | текущий baseline: future return против комиссии/порога |
| `volatility_adjusted_return` | threshold зависит от past/current rolling volatility |
| `triple_barrier` | upper/lower barriers по past volatility и vertical horizon |
| `neutral_zone_return` | расширенная HOLD-зона через buy/sell threshold multipliers |

Для volatility-adjusted и triple-barrier volatility считается только по прошлым/текущим свечам, известным на момент `t`. Future high/low в triple-barrier используются только для target label, не как features.

## 5. Continuous past-only features

Добавлены continuous features в:

```text
ml/src/nlp/action_features.py
```

Feature groups:

- returns over `1,3,6,12,24` candles;
- rolling volatility over `8,16,32`;
- candle body/range/shadows;
- close position inside candle;
- volume ratio and volume z-score vs past rolling volume;
- ATR-like rolling range proxy;
- EMA distance;
- hour/day-of-week cyclic encoding;
- large time gap flag.

Все признаки строятся из текущей или прошлой свечи. Для sample на `t` max feature timestamp <= `t`.

Стандартизация выполняется только по `inner_train` sample indices.

## 6. Модели

Новый CLI поддерживает простые модели без нейросетей:

- `logreg`;
- `ridge`;
- `hist_gb`;
- `extra_trees`;
- `lightgbm`, если доступен в окружении.

В quick run использованы:

```text
logreg,hist_gb
```

## 7. Leakage guarantees

Кодовая схема:

- chronological rolling folds;
- outer train split into `inner_train + calibration`;
- classifier fit только на `inner_train`;
- calibration split не входит в classifier fit;
- validation используется только для оценки;
- test не используется;
- clusterer fit только на `inner_train`;
- LM fit только на `inner_train` words;
- continuous feature scaling fit только на `inner_train` sample rows;
- target horizon не выходит за split boundary;
- future high/low используются только в target labels для `triple_barrier`;
- future returns используются только как supervised target.

## 8. Quick rolling validation

Запущен быстрый прогон:

```powershell
python ml\scripts\sber_action_target_feature_research.py `
  --target-modes return_threshold,volatility_adjusted_return `
  --feature-sets lm_regime,continuous_regime,lm_regime_continuous `
  --models logreg,hist_gb `
  --vocab-config shape:gmm:20 `
  --class-weights balanced,action_boost_1.2 `
  --fold-mode rolling `
  --train-size 12000 `
  --val-size 3000 `
  --step-size 3000 `
  --max-folds 2 `
  --calibration-size 2500 `
  --quick `
  --no-test `
  --output-json data/reports/sber_h1_target_feature_research_quick_20260515.json `
  --output-csv data/reports/sber_h1_target_feature_research_quick_20260515.csv
```

Результаты quick run, top configs:

| target | features | model | weight | macro-F1 | worst | BUY F1 | SELL F1 | HOLD F1 | action rate |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| `return_threshold:h1` | `continuous_regime` | `hist_gb` | `action_boost_1.2` | 0.4465 | 0.4382 | 0.4006 | 0.3911 | 0.5479 | 0.6462 |
| `return_threshold:h1` | `lm_regime_continuous` | `logreg` | `action_boost_1.2` | 0.4439 | 0.4380 | 0.3943 | 0.3769 | 0.5605 | 0.6381 |
| `return_threshold:h1` | `lm_regime_continuous` | `hist_gb` | `action_boost_1.2` | 0.4367 | 0.4259 | 0.4027 | 0.3633 | 0.5441 | 0.6449 |
| `return_threshold:h1` | `lm_regime` | `logreg` | `action_boost_1.2` | 0.4298 | 0.4284 | 0.3843 | 0.3391 | 0.5659 | 0.6343 |
| `return_threshold:h1` | `continuous_regime` | `logreg` | `action_boost_1.2` | 0.4293 | 0.4272 | 0.4064 | 0.3696 | 0.5119 | 0.7419 |

Лучший quick config:

```text
return_threshold:h1 + continuous_regime + hist_gb + action_boost_1.2
macro-F1 = 0.4465
```

## 9. Сравнение с текущим baseline

Текущий validation baseline:

```text
shape/gmm_diag/20 + lm_regime + logreg + action_boost_1.2 + argmax
rolling macro-F1 около 0.4238-0.4265
```

Quick result:

```text
continuous_regime + hist_gb + action_boost_1.2
macro-F1 = 0.4465
```

На двух folds это заметно выше baseline-зоны. Но это еще не финальный вывод: quick run использует только 2 folds, поэтому нужен полный локальный прогон на 4 folds.

## 10. Target analysis

В quick run `volatility_adjusted_return` оказался ниже `return_threshold`:

```text
best vol_adj quick row:
vol_adj:h1:w16:k1 + continuous_regime + hist_gb + balanced
macro-F1 = 0.4133
BUY F1   = 0.2262
SELL F1  = 0.2580
HOLD F1  = 0.7558
```

Vol-adjusted target сделал задачу более HOLD-heavy и пока ухудшил BUY/SELL. Это не закрывает идею полностью, потому что в full run нужно проверить другие `vol_k`, `vol_window` и triple-barrier.

## 11. Feature set comparison

Quick run показывает:

- `continuous_regime` уже сам по себе сильнее текущей LM-линии на 2 folds;
- `lm_regime_continuous` с `logreg` тоже сильный, но не лучше best continuous-only;
- `lm_regime` остается рабочим, но уже не выглядит главным источником качества;
- `hist_gb` на continuous features выглядит перспективнее, чем дальнейшая настройка logreg thresholds.

Предварительный вывод: candle-word LM может быть полезным diagnostic/context block, но continuous past-only features сейчас дают более сильный практический сигнал.

## 12. Что улучшилось

- Появился новый research path без test leakage.
- Continuous baseline сразу дал validation-only improvement на quick run.
- Улучшились BUY/SELL F1 относительно старого LM baseline.
- Появились target modes для проверки более устойчивой постановки labels.

## 13. Что не улучшилось

- Volatility-adjusted target в quick параметрах ухудшил macro-F1 и BUY/SELL.
- Не доказано, что LM + continuous стабильно лучше continuous-only.
- Quick run не заменяет полный rolling run.
- Test не используется и не должен использоваться для подтверждения новых candidates.

## 14. Ограничения

- Quick run только на 2 rolling folds.
- `triple_barrier` и `neutral_zone_return` не вошли в quick run.
- `extra_trees` и полный набор class weights не проверены в quick run.
- Нет production artifact.
- Нет trading claims.

## 15. Следующий шаг

Запустить полный validation-only прогон локально:

```powershell
python ml\scripts\sber_action_target_feature_research.py `
  --target-modes return_threshold,volatility_adjusted_return,triple_barrier,neutral_zone_return `
  --feature-sets lm_regime,continuous_regime,lm_regime_continuous `
  --models logreg,hist_gb,extra_trees `
  --vocab-config shape:gmm:20 `
  --class-weights none,balanced,action_boost_1.2 `
  --fold-mode rolling `
  --train-size 12000 `
  --val-size 3000 `
  --step-size 3000 `
  --max-folds 4 `
  --calibration-size 2500 `
  --no-test `
  --output-json data/reports/sber_h1_target_feature_research_full_20260515.json `
  --output-csv data/reports/sber_h1_target_feature_research_full_20260515.csv
```

Если full run подтвердит `continuous_regime + hist_gb` на 4 folds, следующий research step должен быть не test evaluation, а validation-only refinement continuous features/model regularization и target grid. Test уже использован для предыдущего frozen candidate и не должен участвовать в этом выборе.

## 16. Полный локальный 4-fold validation run пользователя

Пользователь локально запустил полный validation-only target/feature research на 4 rolling folds:

```powershell
python ml\scripts\sber_action_target_feature_research.py `
  --target-modes return_threshold,volatility_adjusted_return,triple_barrier,neutral_zone_return `
  --feature-sets lm_regime,continuous_regime,lm_regime_continuous `
  --models logreg,hist_gb,extra_trees `
  --vocab-config shape:gmm:20 `
  --class-weights none,balanced,action_boost_1.2 `
  --fold-mode rolling `
  --train-size 12000 `
  --val-size 3000 `
  --step-size 3000 `
  --max-folds 4 `
  --calibration-size 2500 `
  --no-test `
  --output-json data/reports/sber_h1_target_feature_research_full_20260515.json `
  --output-csv data/reports/sber_h1_target_feature_research_full_20260515.csv
```

Top validation rows:

| rank | target | features | model | weight | macro-F1 | worst | BUY F1 | SELL F1 | HOLD F1 | action rate |
|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|
| 1 | `return_threshold:h1` | `lm_regime_continuous` | `logreg` | `action_boost_1.2` | 0.4406 | 0.4371 | 0.4000 | 0.3342 | 0.5877 | 0.6008 |
| 2 | `return_threshold:h1` | `continuous_regime` | `extra_trees` | `action_boost_1.2` | 0.4344 | 0.4216 | 0.3923 | 0.3225 | 0.5885 | 0.5694 |
| 3 | `return_threshold:h1` | `continuous_regime` | `hist_gb` | `action_boost_1.2` | 0.4313 | 0.4150 | 0.3851 | 0.3585 | 0.5502 | 0.6231 |
| 4 | `neutral_zone:h1:buy1.5:sell1.5` | `lm_regime_continuous` | `logreg` | `balanced` | 0.4293 | 0.4208 | 0.2747 | 0.2964 | 0.7168 | 0.3557 |
| 5 | `triple_barrier:h3:w16:up1:down1` | `continuous_regime` | `extra_trees` | `none` | 0.4289 | 0.4170 | 0.4276 | 0.4441 | 0.4149 | 0.8018 |

Главный вывод full run:

```text
new validation-primary candidate:
return_threshold:h1 + lm_regime_continuous + logreg + action_boost_1.2
macro-F1 = 0.4406
```

Сравнение со старой LM/action-threshold веткой:

```text
old LM validation zone: 0.4238-0.4265
new full-run best:      0.4406
```

Что подтвердилось:

- качество выросло не от очередного threshold tuning, а от связки `LM + continuous + regime`;
- quick-гипотеза `continuous_regime + hist_gb` не подтвердилась как лучший full-run config: на 4 folds она просела до `0.4313`;
- чистые continuous tree baselines полезны, но лучший результат дал `logreg` на объединенных признаках;
- `triple_barrier` интересен по BUY/SELL F1 (`0.4276/0.4441`), но слишком action-heavy (`action_rate=0.8018`) и слаб по HOLD;
- `neutral_zone` лучше сохраняет HOLD (`HOLD F1=0.7168`), но слабее по BUY/SELL.

Test не использовался и не должен использоваться для выбора новых candidates. Следующий шаг - feature ablation, logreg regularization, return-threshold grid и узкая проверка triple-barrier без обращения к test.
