# SBER H1: feature ablation и target grid для нового action-кандидата

## 1. Цель

Цель этапа - закрепить скачок validation-качества от старой LM/action-threshold зоны `0.4238-0.4265` к новому full-run результату `0.4406` и понять, что именно дает прирост:

- LM-признаки;
- continuous past-only признаки;
- session/time блоки;
- target threshold;
- регуляризация `logreg`;
- модель.

Test split не используется.

## 2. Почему full-run изменил направление

Полный локальный 4-fold validation run пользователя показал новый лучший config:

```text
return_threshold:h1 + lm_regime_continuous + logreg + action_boost_1.2

macro-F1:    0.4406
worst:       0.4371
BUY F1:      0.4000
SELL F1:     0.3342
HOLD F1:     0.5877
action_rate: 0.6008
```

Это заметно выше старой зоны `0.4238-0.4265`. Поэтому дальнейший research должен идти не через старые thresholds, а через ablation, target grid и regularization.

## 3. Новый best validation candidate

Текущий validation-primary candidate:

```text
target:       return_threshold:h1
features:     lm_regime_continuous
model:        logreg
class_weight: action_boost_1.2
decision:     argmax
```

Test `0.4055` относится к предыдущему frozen candidate и не используется здесь.

## 4. Feature ablation

Добавлены feature set modes:

```text
continuous_regime
lm_regime
lm_regime_continuous
continuous_no_session
continuous_no_volume
continuous_no_returns
continuous_no_volatility
continuous_no_candle_shape
lm_regime_continuous_no_lm
lm_regime_continuous_no_session
lm_regime_continuous_no_volume
lm_regime_continuous_no_returns
lm_regime_continuous_no_volatility
lm_regime_continuous_no_candle_shape
```

Трактовка:

- `lm_regime_continuous_no_lm` - continuous-only baseline под тем же именованием ablation;
- `*_no_session` удаляет time/session continuous features и session regime one-hot;
- `*_no_volume` удаляет volume features;
- `*_no_returns` удаляет return/EMA-distance features и trend regime one-hot;
- `*_no_volatility` удаляет volatility/range features и volatility regime one-hot;
- `*_no_candle_shape` удаляет текущую форму свечи из continuous block.

## 5. Logreg tuning

Добавлены CLI-параметры:

```text
--logreg-c-values
--logreg-penalties
--logreg-solvers
```

Невалидные пары solver/penalty отбрасываются:

- `l1`: `liblinear`, `saga`;
- `l2`: `lbfgs`, `liblinear`, `saga`.

Для `l1` в JSON сохраняется `coefficient_sparsity`, если модель имеет `coef_`.

## 6. Return-threshold grid

Добавлен:

```text
--return-threshold-mults
```

Target label теперь имеет вид:

```text
return_threshold:h1:m1.25
```

Multiplier реально меняет threshold вокруг `2 * commission`, включая значения ниже `1.0`. Это исправлено отдельно, потому что старый helper принудительно держал minimum на `2 * commission`.

## 7. Triple-barrier mini-grid

Добавлены удобные параметры:

```text
--barrier-horizons
--barrier-vol-windows
--barrier-k-values
```

`--barrier-k-values` задает симметричный grid `up_k=down_k`, например `1.0,1.25,1.5`.

Цель mini-grid - найти менее action-heavy triple-barrier, не выбирая target только по BUY/SELL F1.

## 8. Tree model narrow tuning

Добавлены compact grids:

```text
--hist-gb-max-iter
--hist-gb-learning-rates
--hist-gb-max-leaf-nodes
--hist-gb-l2
--extra-trees-n-estimators
--extra-trees-max-depths
--extra-trees-min-samples-leaf
--extra-trees-max-features
```

Для `hist_gb` class weights передаются через `sample_weight`, потому что sklearn `HistGradientBoostingClassifier` не принимает `class_weight` напрямую в текущем factory.

## 9. Quick ablation/logreg run

Запущен quick run:

```powershell
python ml\scripts\sber_action_target_feature_research.py `
  --target-modes return_threshold `
  --feature-sets lm_regime_continuous,lm_regime_continuous_no_lm,lm_regime_continuous_no_session,continuous_regime `
  --models logreg `
  --vocab-config shape:gmm:20 `
  --class-weights action_boost_1.2 `
  --return-threshold-mults 0.75,1.0,1.25 `
  --action-horizons 1,2 `
  --logreg-c-values 0.3,1.0,3.0 `
  --fold-mode rolling `
  --train-size 12000 `
  --val-size 3000 `
  --step-size 3000 `
  --max-folds 2 `
  --calibration-size 2500 `
  --no-test `
  --output-json data/reports/sber_h1_feature_ablation_quick_20260515.json `
  --output-csv data/reports/sber_h1_feature_ablation_quick_20260515.csv
```

Top quick rows:

| target | features | model | macro-F1 | worst | BUY F1 | SELL F1 | HOLD F1 | action rate |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `return_threshold:h1:m1.25` | `lm_regime_continuous_no_lm` | `logreg:C=0.3:l2:lbfgs` | 0.4517 | 0.4462 | 0.3505 | 0.3527 | 0.6520 | 0.5174 |
| `return_threshold:h1:m1.25` | `continuous_regime` | `logreg:C=0.3:l2:lbfgs` | 0.4517 | 0.4462 | 0.3505 | 0.3527 | 0.6520 | 0.5174 |
| `return_threshold:h1:m1.25` | `lm_regime_continuous` | `logreg:C=3.0:l2:lbfgs` | 0.4499 | 0.4443 | 0.3594 | 0.3412 | 0.6491 | 0.5013 |
| `return_threshold:h1:m1.0` | `lm_regime_continuous` | `logreg:C=1.0:l2:lbfgs` | 0.4439 | 0.4380 | 0.3943 | 0.3769 | 0.5605 | 0.6381 |
| `return_threshold:h1:m1.25` | `lm_regime_continuous_no_session` | `logreg:C=3.0:l2:lbfgs` | 0.4372 | 0.4352 | 0.3497 | 0.3116 | 0.6502 | 0.4874 |

Quick interpretation:

- На 2 folds `m1.25` резко улучшает macro-F1 за счет более сильного HOLD и более умеренного action rate.
- `lm_regime_continuous_no_lm` совпадает с `continuous_regime`; в quick run это лучше полного LM+continuous.
- Это не отменяет full-run best `0.4406`, потому что quick run только на 2 folds.
- Session/time removal заметно ухудшает quality, значит time/session блок действительно важен.

## 10. Сравнение с baseline

```text
old LM baseline:       0.4238-0.4265
previous full-run best: 0.4406
quick ablation best:    0.4517
```

Quick ablation best нельзя считать новым frozen candidate до полного 4-fold validation.

## 11. Что реально дало прирост

По текущим evidence:

- full run: прирост дала комбинация `LM + continuous + regime`;
- quick ablation: возможно, значительную часть качества дает continuous/time/session block и target threshold `m1.25`;
- `C=0.3` выглядит чуть лучше `C=1.0/3.0` на quick top row;
- удаление session ухудшает результат, значит session не является мусорным блоком.

## 12. Что не подтвердилось

- Пока не доказано, что LM обязателен после добавления continuous признаков.
- Пока не доказано, что `m1.25` сохранит преимущество на 4 folds.
- Tree tuning и triple-barrier mini-grid еще не запущены в этом проходе.

## 13. Ограничения

- Quick ablation run использует только 2 rolling folds.
- Test не используется.
- Нет production artifact.
- Нет trading claims.

## 14. Команды для локального full run

### Full ablation/logreg run

```powershell
python ml\scripts\sber_action_target_feature_research.py `
  --target-modes return_threshold `
  --feature-sets lm_regime_continuous,lm_regime_continuous_no_lm,lm_regime_continuous_no_session,lm_regime_continuous_no_volume,lm_regime_continuous_no_returns,lm_regime_continuous_no_volatility,continuous_regime,lm_regime `
  --models logreg `
  --vocab-config shape:gmm:20 `
  --class-weights action_boost_1.2 `
  --return-threshold-mults 0.75,1.0,1.25,1.5 `
  --action-horizons 1,2,3 `
  --logreg-c-values 0.1,0.3,1.0,3.0,10.0 `
  --fold-mode rolling `
  --train-size 12000 `
  --val-size 3000 `
  --step-size 3000 `
  --max-folds 4 `
  --calibration-size 2500 `
  --no-test `
  --output-json data/reports/sber_h1_feature_ablation_full_20260515.json `
  --output-csv data/reports/sber_h1_feature_ablation_full_20260515.csv
```

### Triple-barrier run

```powershell
python ml\scripts\sber_action_target_feature_research.py `
  --target-modes triple_barrier `
  --feature-sets continuous_regime,lm_regime_continuous `
  --models logreg,extra_trees `
  --vocab-config shape:gmm:20 `
  --class-weights none,balanced,action_boost_1.2 `
  --barrier-horizons 3,6 `
  --barrier-vol-windows 16,32 `
  --barrier-k-values 1.0,1.25,1.5 `
  --fold-mode rolling `
  --train-size 12000 `
  --val-size 3000 `
  --step-size 3000 `
  --max-folds 4 `
  --calibration-size 2500 `
  --no-test `
  --output-json data/reports/sber_h1_triple_barrier_grid_20260515.json `
  --output-csv data/reports/sber_h1_triple_barrier_grid_20260515.csv
```

### Tree narrow tuning

```powershell
python ml\scripts\sber_action_target_feature_research.py `
  --target-modes return_threshold `
  --feature-sets continuous_regime,lm_regime_continuous `
  --models hist_gb,extra_trees `
  --vocab-config shape:gmm:20 `
  --class-weights action_boost_1.2 `
  --action-horizons 1 `
  --return-threshold-mults 1.0 `
  --hist-gb-learning-rates 0.03,0.05,0.10 `
  --hist-gb-max-leaf-nodes 15,31 `
  --hist-gb-l2 0.0,0.1 `
  --extra-trees-max-depths none,8,12 `
  --extra-trees-min-samples-leaf 5,10,20 `
  --fold-mode rolling `
  --train-size 12000 `
  --val-size 3000 `
  --step-size 3000 `
  --max-folds 4 `
  --calibration-size 2500 `
  --no-test `
  --output-json data/reports/sber_h1_tree_tuning_20260515.json `
  --output-csv data/reports/sber_h1_tree_tuning_20260515.csv
```

## 15. Следующий шаг

Нужен полный 4-fold validation run. Если `return_threshold:h1:m1.25 + continuous_regime/logreg C=0.3` подтвердится, новый вопрос будет уже не "помогает ли LM", а "какой минимальный past-only feature set стабильно переносится между folds".
