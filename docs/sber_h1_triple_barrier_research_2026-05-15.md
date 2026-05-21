# SBER H1: строгий research triple-barrier target

Дата: 2026-05-15.

## Цель

Цель этого этапа - проверить, является ли `triple_barrier` реальным улучшением постановки `SELL/HOLD/BUY`, а не артефактом сортировки fold-level строк или одного удачного fold. Test split в этом этапе не используется.

## Fold-level vs aggregate metrics

CSV с fold-level строками нельзя использовать для выбора модели. В таком CSV одна верхняя строка после сортировки по `macro_f1` может соответствовать одному удачному fold, например `fold_id=2`.

Для выбора config используются только aggregate metrics по всем folds:

- `mean_macro_f1`;
- `worst_macro_f1`;
- `std_macro_f1`;
- `mean_buy_f1`;
- `mean_sell_f1`;
- `mean_hold_f1`;
- `mean_action_rate`.

Новый CLI сохраняет два CSV:

- folds CSV: диагностика отдельных folds;
- aggregate CSV: таблица для model selection.

В console summary явно печатается: `AGGREGATE RESULTS, not fold-level rows`.

## Инкорпорированные локальные результаты

Пользователь локально выполнил полные validation-only прогоны на 4 rolling folds.

Старый LM/action baseline держался в зоне:

```text
macro-F1: 0.4238-0.4265
```

Лучший return-threshold ablation:

```text
target:       return_threshold:h1:m1
features:     lm_regime_continuous_no_volatility
model:        logreg:C=1:penalty=l2:solver=lbfgs
class_weight: action_boost_1.2
mean macro-F1: 0.4419
worst:         0.4316
BUY F1:        0.3966
SELL F1:       0.3391
HOLD F1:       0.5899
action_rate:   0.6004
```

Лучший return-threshold tree config:

```text
target:       return_threshold:h1:m1
features:     continuous_regime
model:        extra_trees:depth=none:leaf=5:maxfeat=sqrt
class_weight: action_boost_1.2
mean macro-F1: 0.4405
worst:         0.4304
BUY F1:        0.4097
SELL F1:       0.3460
HOLD F1:       0.5660
action_rate:   0.6449
```

Лучший aggregate triple-barrier config из локального прогона:

```text
target:       triple_barrier:h3:w16:up1.25:down1.5
features:     continuous_regime
model:        extra_trees:depth=none:leaf=5:maxfeat=sqrt
class_weight: balanced
mean macro-F1: 0.4589
worst:         0.4333
BUY F1:        0.4257
SELL F1:       0.3981
HOLD F1:       0.5529
action_rate:   0.6955
```

Fold-level top row около `0.4886` на `fold_id=2` является только диагностикой. Это не aggregate result и не selection metric.

## Полный focused triple-barrier full run пользователя

Пользователь локально запустил focused validation-only triple-barrier grid. Выбор выполнялся только по aggregate CSV, не по fold-level строкам.

Команда запуска:

```powershell
python ml\scripts\sber_triple_barrier_research.py `
  --barrier-horizons 3,4,6 `
  --barrier-vol-windows 12,16,24,32 `
  --barrier-up-k-values 1.0,1.25,1.5 `
  --barrier-down-k-values 1.25,1.5,1.75,2.0 `
  --feature-sets continuous_regime,continuous_no_session,continuous_no_volatility,lm_regime_continuous `
  --models extra_trees `
  --class-weights balanced,none `
  --extra-trees-n-estimators 300 `
  --extra-trees-min-samples-leaf 5,10,20 `
  --extra-trees-max-depths none,12 `
  --extra-trees-max-features sqrt,0.7 `
  --fold-mode rolling `
  --train-size 12000 `
  --val-size 3000 `
  --step-size 3000 `
  --max-folds 4 `
  --calibration-size 2500 `
  --no-test `
  --include-target-audit `
  --include-economic-sanity `
  --output-json data/reports/sber_h1_triple_barrier_research_full_20260515.json `
  --output-csv data/reports/sber_h1_triple_barrier_research_full_folds_20260515.csv `
  --output-aggregate-csv data/reports/sber_h1_triple_barrier_research_full_aggregate_20260515.csv
```

Ожидаемые файлы отчета:

- `data/reports/sber_h1_triple_barrier_research_full_20260515.json`;
- `data/reports/sber_h1_triple_barrier_research_full_folds_20260515.csv`;
- `data/reports/sber_h1_triple_barrier_research_full_aggregate_20260515.csv`.

Новый лучший validation-only aggregate config:

```text
target:       triple_barrier:h3:w12:up1.25:down1.25
features:     continuous_regime
model:        extra_trees:depth=none:leaf=20:maxfeat=sqrt
class_weight: none
random_state: 42
n_folds:      4

mean macro-F1:      0.4694586125565165
std macro-F1:       0.009063228583320601
worst macro-F1:     0.4547935587339986
mean accuracy:      0.4720161834120027
balanced accuracy:  0.4697147728103302
BUY F1:             0.4064398078493485
SELL F1:            0.43872171796117804
HOLD F1:            0.563214311859023
BUY/SELL hmean:     0.42120448193929483
action rate:        0.6725387727579231
hold rate:          0.3274612272420769
prediction BUY:     0.30917060013486175
prediction SELL:    0.3633681726230614
prediction HOLD:    0.3274612272420769
```

Сравнение с предыдущими validation-only результатами:

```text
old LM/action baseline:       0.4238-0.4265
return-threshold best:        0.4419
previous triple-barrier best:  0.4589
new triple-barrier best:       0.4695
```

Интерпретация:

- focused triple-barrier стал новым лучшим validation-only направлением;
- улучшение пришло не от очередного threshold tuning, а от другой постановки target;
- `HOLD F1` остается выше 0.50, то есть target не полностью разваливает нейтральный класс;
- `action_rate` высокий, но пока находится в заранее допустимой исследовательской зоне `0.55-0.75`.

Риски:

- результат пока только для `random_state=42`;
- это validation-only результат, test не использовался;
- fold-level строки не являются selection metric;
- action rate около `0.67` требует отдельной проверки экономического смысла и устойчивости;
- перед frozen candidate нужен seed robustness для ExtraTrees и повторная проверка target/economic audit.

## Target audit

Для `triple_barrier` добавлен audit по каждому target config/fold:

- `target_distribution_train`;
- `target_distribution_calibration`;
- `target_distribution_val`;
- `share_upper_first`;
- `share_lower_first`;
- `share_vertical_timeout`;
- `share_ambiguous`;
- `mean_time_to_barrier`;
- `median_time_to_barrier`;
- `time_to_barrier` по label;
- mean/median future return по label;
- MFE/MAE по label.

MFE/MAE считаются по future high/low внутри target horizon и используются только для target/evaluation diagnostics, не как признаки модели.

## Leakage/alignment audit

Для `--dump-target-audit-samples N` CLI сохраняет примеры:

- `sample_idx`;
- `decision_time`;
- `close_t`;
- `past_vol_t`;
- `upper_barrier`;
- `lower_barrier`;
- future highs/lows и timestamps;
- `label`;
- `outcome`;
- `time_to_barrier`;
- `max_feature_time`.

Проверяемое правило:

```text
max_feature_time <= decision_time
target uses only candles t+1 ... t+horizon
features use only current/past candles
```

Future high/low допустимы только в target label и diagnostics. Test split не используется.

## Focused triple-barrier research

Подготовлен отдельный CLI:

```text
ml/scripts/sber_triple_barrier_research.py
```

Он является thin wrapper над `sber_action_target_feature_research.py` и по умолчанию inject-ит `--target-modes triple_barrier`.

Focused grid вокруг текущего aggregate-best:

```text
barrier_horizon: 3,4,6
vol_window:      12,16,24,32
up_k:            1.0,1.25,1.5
down_k:          1.25,1.5,1.75,2.0
features:        continuous_regime, continuous_no_session, continuous_no_volatility, lm_regime_continuous
model:           extra_trees
```

Selection выполняется только по aggregate validation metrics.

## Economic sanity

Добавлены validation-only diagnostics:

- mean/median realized return by prediction;
- BUY directional hit rate;
- SELL directional hit rate;
- HOLD mean absolute future return;
- predicted BUY upper-barrier hit rate;
- predicted SELL lower-barrier hit rate;
- predicted action barrier hit rate.

Это не trading backtest и не утверждение о прибыльности. Метрики нужны только для проверки, имеет ли target/model экономически осмысленное направление.

## Лучший validation-only candidate на текущий момент

Пока primary direction:

```text
triple_barrier:h3:w16:up1.25:down1.5
features = continuous_regime
model = extra_trees:depth=none:leaf=5:maxfeat=sqrt
class_weight = balanced
```

Он лучше return-threshold ветки по aggregate macro-F1, но имеет высокий `action_rate` около `0.6955`. Перед любым frozen candidate нужны focused grid, seed robustness и target audit.

## Ограничения

- Test split не используется и не должен использоваться для нового выбора.
- Triple-barrier меняет сам target, поэтому результат нельзя напрямую трактовать как улучшение старого return-threshold label.
- Высокий action rate может означать слишком агрессивную постановку target.
- Fold-level maxima нельзя использовать для выбора.
- Economic sanity diagnostics не заменяют trading backtest.

## Следующий шаг

1. Запустить focused triple-barrier full run и анализировать только aggregate CSV.
2. Запустить ExtraTrees seed robustness вокруг `h3/w16/up1.25/down1.5`.
3. Проверить, сохраняется ли преимущество при разумном `HOLD F1 >= 0.50` и `action_rate` в зоне `0.55-0.75`.
4. Только после этого решать, становится ли triple-barrier новым primary target direction.
