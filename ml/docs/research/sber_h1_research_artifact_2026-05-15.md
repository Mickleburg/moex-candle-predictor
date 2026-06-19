# SBER H1 research artifact

## Цель

Цель этого этапа - превратить frozen validation-only triple-barrier candidate в локальный research artifact bundle, который можно использовать для интеграционной проверки:

```text
candle_batch JSON -> ml_prediction JSON
```

Это не production trading artifact, не live trading и не final test evaluation.

## Frozen candidate

```text
target:       triple_barrier:h3:w12:up1.25:down1.25
features:     continuous_regime
model:        extra_trees:depth=none:leaf=20:maxfeat=sqrt
class_weight: none
n_estimators: 300
random_state: 42
```

Seed robustness:

```text
mean macro-F1:       0.4685
previous seed=42:    0.4695
worst seed:          0.4676
best seed:           0.4695
std across seeds:    0.0008
std across folds:    0.0087
worst fold:          0.4522
BUY F1:              0.4044
SELL F1:             0.4377
HOLD F1:             0.5634
action rate:         0.6708
```

## Training protocol

Для artifact выбран консервативный research protocol:

```text
training_protocol: development_only
development data: first 85% chronological rows
untouched tail: last 15%
test tuning: no
final test evaluation: no
```

Artifact нужен для contract inference и integration testing. Его создание не добавляет нового утверждения о качестве модели.

## Artifact contents

Локальная директория:

```text
ml/artifacts/research_triple_barrier_sber_h1/
```

Файлы:

```text
model.pkl
feature_config.json
target_config.json
metadata.json
label_mapping.json
schema_version.json
feature_columns.json
training_summary.json
```

`model.pkl` и весь bundle считаются локально генерируемым research artifact и не коммитятся в git.

## Команда обучения

```powershell
python ml\scripts\train_research_artifact.py `
  --ticker SBER `
  --timeframe 1H `
  --target-mode triple_barrier `
  --barrier-horizon 3 `
  --barrier-vol-window 12 `
  --barrier-up-k 1.25 `
  --barrier-down-k 1.25 `
  --feature-set continuous_regime `
  --model extra_trees `
  --n-estimators 300 `
  --min-samples-leaf 20 `
  --max-depth none `
  --max-features sqrt `
  --class-weight none `
  --random-state 42 `
  --training-protocol development_only `
  --output-dir ml\artifacts\research_triple_barrier_sber_h1
```

## Inference path

Без artifact:

```powershell
python ml\scripts\predict_from_json.py `
  --input-json contracts\examples\candle_batch.example.json `
  --output-json data\reports\ml_prediction_example_artifact_missing.json
```

Результат:

```text
diagnostics.artifact_missing=true
probabilities = {buy: 0, hold: 1, sell: 0}
confidence = 0
```

С artifact:

```powershell
python ml\scripts\predict_from_json.py `
  --input-json contracts\examples\candle_batch.example.json `
  --artifact-dir ml\artifacts\research_triple_barrier_sber_h1 `
  --output-json data\reports\ml_prediction_example_with_artifact.json
```

Результат:

```text
diagnostics.artifact_missing=false
diagnostics.is_production=false
probabilities = predict_proba over buy/hold/sell
```

## JSON contract

Вход:

```text
contracts/candle_batch.schema.json
```

Выход:

```text
contracts/ml_prediction.schema.json
```

Внутренние классы отображаются так:

```text
SELL -> sell
HOLD -> hold
BUY  -> buy
```

## Ограничения

- Artifact research-only.
- Probabilities не калиброваны.
- Target является triple-barrier action target, а не прямым прогнозом цены или доходности.
- Final test не запускался.
- Backtest и paper trading не выполнялись.
- Risk manager, aggregator, LLM и execution не реализуются в этом проходе.
- Нельзя использовать этот artifact как основание для live trading.

## Checks

Минимальные проверки:

```powershell
python -m compileall -q ml\src ml\scripts ml\test_smoke.py scripts\validate_contracts.py
python ml\test_smoke.py
python scripts\validate_contracts.py
python ml\scripts\train_research_artifact.py ... --output-dir ml\artifacts\research_triple_barrier_sber_h1
python ml\scripts\predict_from_json.py --input-json contracts\examples\candle_batch.example.json --artifact-dir ml\artifacts\research_triple_barrier_sber_h1 --output-json data\reports\ml_prediction_example_with_artifact.json
python ml\scripts\predict_from_json.py --input-json contracts\examples\candle_batch.example.json --output-json data\reports\ml_prediction_example_artifact_missing.json
git diff --check
```

## Следующий шаг

Следующий исследовательский шаг - probability calibration или prediction logging для будущего backtest/paper-ready контура. До этого artifact остается только contract-compatible research integration artifact.
