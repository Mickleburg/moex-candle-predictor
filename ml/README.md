# ML Block

`ml/` - текущий основной рабочий research-блок проекта. Он отвечает за загрузку свечей, очистку, построение признаков, candle-language эксперименты, action target research и legacy FastAPI inference path.

## Что уже есть

- Legacy production inference path:
  - `ml/artifacts/model.pkl`;
  - `ml/artifacts/tokenizer.pkl`;
  - `ml/artifacts/metadata.json`;
  - `ml/src/service/api.py`;
  - `ml/src/service/predictor.py`.
- Candle-language research:
  - свеча как candle word;
  - sentence windows;
  - TF-IDF/cooccurrence/SVD;
  - n-gram/backoff word LM;
  - next-word continuation metrics.
- Action classification research:
  - return-threshold target;
  - LM-derived features;
  - continuous past-only features;
  - nested threshold calibration;
  - triple-barrier target.
- Diagnostics:
  - candle accounting;
  - calendar/raw coverage audit;
  - walk-forward validation;
  - target audit;
  - leakage/alignment smoke checks.

## Research-направления

- Legacy baseline для `SELL/HOLD/BUY`.
- Candle-word LM и sequence continuation.
- Return-threshold action target.
- Triple-barrier action target.
- Continuous past-only feature baseline.
- LM + continuous feature combinations.

## Текущий лучший validation-only research candidate

```text
target:       triple_barrier:h3:w12:up1.25:down1.25
features:     continuous_regime
model:        extra_trees:depth=none:leaf=20:maxfeat=sqrt
class_weight: none
mean macro-F1: 0.4695
```

Это не production artifact и не разрешение на торговлю. Test split не использовался для выбора этого candidate.

Seed robustness для этого candidate проведен на seeds `7,13,21,42,100`:

```text
mean macro-F1 over seeds: 0.4685
worst seed macro-F1:     0.4676
worst fold macro-F1:     0.4522
BUY F1:                  0.4044
SELL F1:                 0.4377
HOLD F1:                 0.5634
action rate:             0.6708
```

По этой проверке candidate можно считать frozen research candidate для следующего этапа artifact bundle protocol. Сам artifact bundle пока не создан, а `predict_from_json.py` по-прежнему возвращает `diagnostics.artifact_missing=true`.

После seed robustness перед production research artifact все еще нужны:

1. frozen candidate protocol;
2. одна честная final evaluation, если protocol разрешает;
3. backtest и paper trading;
4. явный risk layer.

## Команды проверки

```powershell
python -m compileall -q ml\src ml\scripts ml\test_smoke.py
python ml\test_smoke.py
```

Общие architecture contracts проверяются из корня проекта:

```powershell
python scripts\validate_contracts.py
```

## ML prediction JSON contract

ML-блок умеет принимать `candle_batch` JSON и записывать `ml_prediction` JSON:

```powershell
python ml\scripts\predict_from_json.py `
  --input-json contracts\examples\candle_batch.example.json `
  --output-json data\reports\ml_prediction_example.json
```

Без `--artifact-dir` команда сохраняет честный placeholder с `diagnostics.artifact_missing=true`.

Локальный research artifact для frozen triple-barrier candidate можно собрать так:

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

После этого можно получить real `predict_proba` probabilities по тому же JSON contract:

```powershell
python ml\scripts\predict_from_json.py `
  --input-json contracts\examples\candle_batch.example.json `
  --artifact-dir ml\artifacts\research_triple_barrier_sber_h1 `
  --output-json data\reports\ml_prediction_example_with_artifact.json
```

Входной контракт:

```text
contracts/candle_batch.schema.json
```

Выходной контракт:

```text
contracts/ml_prediction.schema.json
```

Research artifact остается `is_production=false`: это интеграционный artifact для проверки ML JSON I/O, а не trading artifact. Probabilities пока не калиброваны, target является triple-barrier action target, а не direct price forecast. Binary artifact bundle генерируется локально и не должен попадать в git без отдельного решения.

Подробности:

- `ml/docs/ml_prediction_contract_2026-05-15.md`;
- `ml/docs/research/sber_h1_research_artifact_2026-05-15.md`.

## Документы

- `ml/docs/research/` - SBER H1 research reports.
- `ml/docs/README.md` - карта ML-документации.
