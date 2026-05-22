# ML prediction JSON contract

Дата: 2026-05-15.

## Цель

ML-блок должен быть интеграционным компонентом будущего агента:

```text
candle_batch JSON -> ml_prediction JSON
```

Этот слой не включает торговлю, risk approval, aggregator, LLM или execution. Он только нормализует входные свечи, подготавливает данные для ML inference и возвращает JSON по контракту.

## Вход

Вход соответствует:

```text
contracts/candle_batch.schema.json
```

Минимальная структура:

```json
{
  "ticker": "SBER",
  "timeframe": "1H",
  "candles": [
    {
      "begin": "2026-05-15T15:00:00+03:00",
      "open": 300.0,
      "high": 302.0,
      "low": 299.0,
      "close": 301.5,
      "volume": 123456
    }
  ]
}
```

ML contract layer проверяет:

- обязательные поля `ticker`, `timeframe`, `candles`;
- обязательные поля свечи `begin/open/high/low/close/volume`;
- корректность timestamps;
- сортировку по `begin`;
- отсутствие duplicate `begin`;
- отсутствие mixed ticker/timeframe;
- finite OHLCV;
- базовую OHLC-согласованность.

## Выход

Выход соответствует:

```text
contracts/ml_prediction.schema.json
```

Ключевые поля:

- `ticker`;
- `timeframe`;
- `as_of`;
- `model_version`;
- `model_family`;
- `target`;
- `probabilities.buy/hold/sell`;
- `confidence`;
- `expected_return`;
- `diagnostics`.

Для action classification внутренние классы отображаются так:

```text
SELL -> sell
HOLD -> hold
BUY  -> buy
```

## Текущий research-default

Текущий лучший validation-only research candidate:

```text
target:       triple_barrier:h3:w12:up1.25:down1.25
features:     continuous_regime
model:        extra_trees:depth=none:leaf=20:maxfeat=sqrt
class_weight: none
mean macro-F1: 0.4695
```

Это не production artifact. Test split не используется для выбора новых candidates. Перед production artifact нужны seed robustness, frozen evaluation protocol, backtest и paper trading.

## Artifact missing mode

На момент этого документа fitted production artifact для `triple_barrier_extra_trees` не зафиксирован. Поэтому `ml/scripts/predict_from_json.py` возвращает валидный `ml_prediction` JSON с:

```json
{
  "probabilities": {
    "buy": 0.0,
    "hold": 1.0,
    "sell": 0.0
  },
  "confidence": 0.0,
  "diagnostics": {
    "artifact_missing": true,
    "is_production": false
  }
}
```

Это не прогноз. Это честный contract-compatible placeholder, который позволяет интеграционным блокам работать с форматом ответа без имитации качества модели.

## CLI

```powershell
python ml\scripts\predict_from_json.py `
  --input-json contracts\examples\candle_batch.example.json `
  --output-json data\reports\ml_prediction_example.json
```

## Будущий artifact bundle

Для настоящего `triple_barrier_extra_trees` inference нужен bundle:

```text
model.pkl
feature_pipeline.pkl или feature_config.json
target_config.json
metadata.json
label_mapping.json
calibration.json, если появится
schema_version
```

Минимальная metadata:

```json
{
  "model_family": "triple_barrier_extra_trees",
  "target": "triple_barrier:h3:w12:up1.25:down1.25",
  "feature_set": "continuous_regime",
  "class_weight": "none",
  "validation_macro_f1": 0.4695,
  "is_production": false
}
```

## Проверки

```powershell
python -m compileall -q ml\src ml\scripts ml\test_smoke.py
python ml\test_smoke.py
python scripts\validate_contracts.py
python ml\scripts\predict_from_json.py `
  --input-json contracts\examples\candle_batch.example.json `
  --output-json data\reports\ml_prediction_example.json
```
