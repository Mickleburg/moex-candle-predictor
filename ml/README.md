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

Перед production research artifact нужны:

1. seed robustness для ExtraTrees;
2. frozen candidate protocol;
3. одна честная final evaluation, если protocol разрешает;
4. backtest и paper trading;
5. явный risk layer.

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

Входной контракт:

```text
contracts/candle_batch.schema.json
```

Выходной контракт:

```text
contracts/ml_prediction.schema.json
```

Текущий лучший research candidate (`triple_barrier:h3:w12:up1.25:down1.25`) пока не имеет fitted production artifact bundle. Поэтому CLI возвращает валидный JSON в режиме `diagnostics.artifact_missing=true`, а не настоящий прогноз.

Подробности: `ml/docs/ml_prediction_contract_2026-05-15.md`.

## Документы

- `ml/docs/research/` - SBER H1 research reports.
- `ml/docs/README.md` - карта ML-документации.
