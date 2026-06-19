# SBER H1: triple-barrier seed robustness

## Цель

Цель прохода - проверить, является ли текущий лучший validation-only triple-barrier candidate устойчивым к `random_state` ExtraTrees, или результат `random_state=42` был удачным случайным запуском.

Этот проход не использует test split, не делает final evaluation, не создает production artifact и не является торговым backtest.

## Проверяемый candidate

```text
target:       triple_barrier:h3:w12:up1.25:down1.25
features:     continuous_regime
model:        extra_trees:depth=none:leaf=20:maxfeat=sqrt
class_weight: none
random_states: 7,13,21,42,100
fold_mode:    rolling
n_folds:      4
```

Предыдущий single-seed результат для `random_state=42`:

```text
mean macro-F1: 0.4695
worst fold:    0.4548
BUY F1:        0.4064
SELL F1:       0.4387
HOLD F1:       0.5632
action rate:   0.6725
```

## Почему нужен seed robustness

`ExtraTreesClassifier` зависит от случайности: bootstrap/feature randomness и структура деревьев могут менять прогнозы. Перед frozen research candidate и будущим artifact bundle нужно проверить, что качество не держится на одном удачном seed.

## Команда запуска

```powershell
python ml\scripts\sber_triple_barrier_research.py `
  --barrier-horizons 3 `
  --barrier-vol-windows 12 `
  --barrier-up-k-values 1.25 `
  --barrier-down-k-values 1.25 `
  --feature-sets continuous_regime `
  --models extra_trees `
  --class-weights none `
  --extra-trees-n-estimators 300 `
  --extra-trees-min-samples-leaf 20 `
  --extra-trees-max-depths none `
  --extra-trees-max-features sqrt `
  --random-states 7,13,21,42,100 `
  --fold-mode rolling `
  --train-size 12000 `
  --val-size 3000 `
  --step-size 3000 `
  --max-folds 4 `
  --calibration-size 2500 `
  --no-test `
  --include-target-audit `
  --include-economic-sanity `
  --output-json data/reports/sber_h1_triple_barrier_best_seed_robustness_20260515.json `
  --output-csv data/reports/sber_h1_triple_barrier_best_seed_robustness_folds_20260515.csv `
  --output-aggregate-csv data/reports/sber_h1_triple_barrier_best_seed_robustness_aggregate_20260515.csv
```

Отчеты:

```text
data/reports/sber_h1_triple_barrier_best_seed_robustness_20260515.json
data/reports/sber_h1_triple_barrier_best_seed_robustness_folds_20260515.csv
data/reports/sber_h1_triple_barrier_best_seed_robustness_aggregate_20260515.csv
```

## Aggregate results

Важно: selection делается только по aggregate metrics. Fold-level rows используются только для диагностики.

```text
mean macro-F1:       0.4685248640
std macro-F1:        0.0089210587
worst macro-F1:      0.4522259133
std across folds:    0.0087174892
std across seeds:    0.0007977174
BUY F1:              0.4044161506
SELL F1:             0.4377439752
HOLD F1:             0.5634144663
BUY/SELL hmean:      0.4197061175
action rate:         0.6708024275
prediction BUY:      0.3070296696
prediction SELL:     0.3637727579
prediction HOLD:     0.3291975725
```

## Seed-level results

```text
seed 7:   mean macro-F1 = 0.4694429582; min fold = 0.4545500917; max fold = 0.4820638976
seed 13:  mean macro-F1 = 0.4677888090; min fold = 0.4542211617; max fold = 0.4759173197
seed 21:  mean macro-F1 = 0.4683560163; min fold = 0.4562478378; max fold = 0.4755905641
seed 42:  mean macro-F1 = 0.4694586126; min fold = 0.4547935587; max fold = 0.4794213670
seed 100: mean macro-F1 = 0.4675779240; min fold = 0.4522259133; max fold = 0.4777272394
```

```text
worst seed macro-F1: 0.4675779240
best seed macro-F1:  0.4694586126
seed spread:         0.0018806886
```

## Fold-level stability

```text
fold 1: mean macro-F1 = 0.4717836637
fold 2: mean macro-F1 = 0.4781440775
fold 3: mean macro-F1 = 0.4544077126
fold 4: mean macro-F1 = 0.4697640021
```

`std_across_folds = 0.00872`, а `std_across_seeds = 0.00080`. Значит, variance по временным folds примерно на порядок выше, чем variance по seeds. Модель чувствительнее к рыночному режиму validation interval, чем к initialization ExtraTrees.

## Target audit

Среднее распределение target на validation folds:

```text
BUY:  0.3194
SELL: 0.3365
HOLD: 0.3442
```

Средние barrier outcomes на validation:

```text
upper first:      0.3194
lower first:      0.3365
vertical timeout: 0.2951
ambiguous:        0.0491
mean time:        2.03 candles
median time:      2 candles
```

Target остается достаточно сбалансированным для `BUY/SELL/HOLD`, но action labels суммарно составляют примерно две трети выборки. Это объясняет высокий action rate модели и требует отдельного risk layer в будущем.

## Economic sanity diagnostics

Это validation-only diagnostics, не backtest и не claim о прибыльности.

```text
directional hit rate for predicted BUY:  0.5352
directional hit rate for predicted SELL: 0.5160
predicted BUY upper-hit rate:            0.4140
predicted SELL lower-hit rate:           0.4229
predicted action barrier-hit rate:       0.7723
HOLD mean abs future return:             0.00270
```

Mean realized return by predicted action:

```text
predicted BUY:  +0.000187
predicted SELL: -0.000071
predicted HOLD: +0.000133
```

Санити-метрики выглядят умеренно согласованными с направлением target, но они не учитывают комиссии, исполнение, проскальзывание, position sizing и risk constraints. Для SELL средний future return отрицательный, что лучше, чем в некоторых ранних sanity-runs, но величина мала; экономический смысл нужно проверять отдельно.

## Сравнение с previous random_state=42 result

```text
previous seed=42 mean macro-F1: 0.4695
robustness mean over seeds:     0.4685
delta:                         -0.0009
```

Single-seed result `42` не выглядит случайным выбросом: он находится у верхней границы seed range, но отличается от среднего меньше чем на 0.001 macro-F1.

## Интерпретация

Seed robustness подтверждает устойчивость candidate:

- `mean macro-F1 = 0.4685`, выше порога 0.46;
- `worst seed macro-F1 = 0.4676`, выше порога 0.455;
- `worst fold macro-F1 = 0.4522`, выше желательного порога 0.45;
- `HOLD F1 = 0.5634`, выше порога 0.50;
- `action rate = 0.6708`, внутри разумной validation-only зоны 0.55-0.75;
- seed variance низкая, fold variance доминирует.

Candidate можно считать frozen research candidate для следующего этапа artifact bundle protocol.

## Ограничения

- Test split не использовался.
- Это не production artifact.
- Это не торговая стратегия и не backtest.
- Economic sanity является диагностикой target/model behavior, а не оценкой доходности.
- Fold 3 остается самым слабым временным режимом.
- Перед реальным использованием нужны frozen artifact protocol, контроль схемы признаков, final evaluation по заранее утвержденному protocol, backtest и paper trading.

## Вывод

`triple_barrier:h3:w12:up1.25:down1.25 + continuous_regime + ExtraTrees leaf=20` устойчив по seeds и может быть зафиксирован как frozen research candidate для следующего шага.

## Следующий шаг

Следующий логичный этап - не новый tuning, а artifact bundle protocol:

```text
model.pkl
feature_config.json
target_config.json
metadata.json
label_mapping.json
schema_version
```

До подключения к `predict_from_json.py` artifact должен быть явно помечен как research artifact, не production trading artifact.
