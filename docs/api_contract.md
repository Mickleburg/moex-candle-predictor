# API Contract: ML Inference

Документ описывает фактический HTTP-контракт между backend и ML inference service.

## Endpoints

### `GET /health`

Ответ:

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "2026-04-19T17:29:17.793124",
  "timestamp": "2026-04-22T08:00:00"
}
```

Замечания:

- endpoint всегда отвечает JSON;
- `status` может быть `healthy` или `degraded`;
- сам сервис может стартовать даже без загруженной модели.

### `POST /predict`

Request:

```json
{
  "candles": [
    {
      "begin": "2024-01-01T10:00:00",
      "open": 250.0,
      "high": 251.5,
      "low": 249.0,
      "close": 250.5,
      "volume": 150000,
      "ticker": "SBER",
      "timeframe": "1H"
    }
  ],
  "model_version": null
}
```

Response:

```json
{
  "predicted_token": 4,
  "probabilities": [0.1, 0.05, 0.1, 0.15, 0.3, 0.2, 0.1],
  "action": "buy",
  "confidence": 0.3,
  "model_version": "2026-04-19T17:29:17.793124",
  "ticker": "SBER",
  "timeframe": "1H",
  "timestamp": "2026-04-22T08:00:00",
  "n_candles_used": 32,
  "diagnostics": {
    "window_size": 32,
    "n_classes": 7,
    "horizon": 3
  }
}
```

## Request constraints

- требуется массив `candles`;
- каждая свеча должна содержать `begin`, `open`, `high`, `low`, `close`, `volume`;
- `ticker` и `timeframe` optional;
- по коду inference требует минимум `window_size` свечей.

## Error semantics

- `400` — некорректный request или недостаточно свечей;
- `503` — отсутствуют артефакты модели;
- `500` — внутренняя ошибка preprocessing или модели.

## Shared schemas

Актуальные schema files:

- `shared/schemas/candles_request.json`
- `shared/schemas/prediction_response.json`

Эти схемы описывают форму request/response, но не все runtime-ограничения:

- не выражают минимальный размер окна;
- не выражают требования к порядку свечей;
- не выражают train/serve совместимость артефактов.

## Что backend предполагает дополнительно

Поверх JSON Schema backend ожидает, что:

- свечи относятся к одному ticker и одному timeframe;
- свечи отсортированы от старых к новым;
- ML response возвращает осмысленные `probabilities`, `confidence`, `model_version`;
- `action` ограничен значениями `buy`, `sell`, `hold`.
