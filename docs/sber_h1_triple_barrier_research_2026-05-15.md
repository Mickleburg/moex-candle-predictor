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
