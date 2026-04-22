# API Contract

## Scope

Этот документ фиксирует только текущий подтверждаемый контракт между Go backend и Python ML inference service, плюс фактический raw data contract, на который опирается training/inference путь.

## Source Of Truth

Приоритет источников истины:

1. код;
2. `shared/schemas/` и фактические runtime models;
3. `backend_integration.pdf` как reference, если он не противоречит коду.

Важно:

- `shared/schemas/` покрывают только `POST /predict`;
- `GET /health` и raw candle storage contract схемами не покрыты;
- PDF использует camelCase примеры, но текущий код работает в snake_case.

## Backend <-> ML Contract

### Endpoint: `GET /health`

Назначение:

- проверить доступность ML service;
- определить, загружены ли artifacts;
- получить `model_version`, если она известна.

Фактический response shape по коду ML service:

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "2026-04-19T17:29:17.793124",
  "timestamp": "2026-04-22T08:00:00"
}
```

Поля:

- `status`: `"healthy"` или `"degraded"`
- `model_loaded`: `true` / `false`
- `model_version`: string или `null`
- `timestamp`: ISO datetime

Backend-side semantics:

- `PredictService.Health` ожидает именно snake_case поля;
- при `strict_health_check: true` backend требует `status == healthy` и `model_loaded == true` перед `POST /predict`;
- backend `/health` агрегирует ML health в свой собственный payload, но это отдельный backend contract.

### Endpoint: `POST /predict`

Назначение:

- передать последние свечи;
- получить предсказанный класс, вероятности и action.

#### Request

Фактический request shape:

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

Поля:

- `candles`: массив свечей
- `model_version`: optional string/null

Поля свечи:

- `begin`: обязательный datetime
- `open`: обязательное число
- `high`: обязательное число
- `low`: обязательное число
- `close`: обязательное число
- `volume`: обязательное число
- `ticker`: optional, но backend обычно передает явно
- `timeframe`: optional, но backend обычно передает явно

#### Response

Фактический response shape:

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

Поля:

- `predicted_token`: integer
- `probabilities`: array of float
- `action`: `"buy" | "sell" | "hold"`
- `confidence`: float
- `model_version`: string
- `ticker`: string
- `timeframe`: string
- `timestamp`: ISO datetime
- `n_candles_used`: integer
- `diagnostics`: optional object

## Validation Rules

### Schema-Level

`shared/schemas/candles_request.json` и `shared/schemas/prediction_response.json` задают форму request/response.

Они проверяют:

- наличие обязательных полей;
- базовые JSON types;
- `action` enum в response.

### Runtime-Level

Код добавляет более строгие ограничения, которых нет в JSON Schema:

- свечей должно быть не меньше `window_size`;
- inference ожидает данные в хронологическом порядке;
- request логически предполагает один ticker;
- request логически предполагает один timeframe;
- вход должен быть совместим с Python preprocessing path.

## Minimum Candles

На текущем кодовом пути есть два согласованных источника значения:

- backend config: `ml.min_candles = 32`
- checked-in ML metadata: `L = 32`

Практически это означает:

- backend не отправляет request, если свечей меньше `32`;
- ML predictor также отклонит request, если свечей меньше `window_size`.

Это число нельзя считать навсегда фиксированным для будущих версий без синхронного изменения backend config, artifacts и документации.

## Ordering Requirements

Требование порядка:

- candles должны идти от старых к новым.

Почему это критично:

- backend history validation сортирует batch перед storage;
- inference строит окно из последних `window_size` записей;
- нарушение порядка меняет смысл признаков и таргета.

## Single Ticker / Single Timeframe Assumptions

Текущий код и docs опираются на такие предположения:

- один inference request = один ticker;
- один inference request = один timeframe;
- один training run фактически использует первый ticker и первый timeframe из ML config.

Backend explicitly проверяет mixed ticker/timeframe при работе с `models.Candle`.

## Error Handling Semantics

### `400 Bad Request`

Типичные причины:

- меньше минимального числа свечей;
- неверный request body;
- missing required fields;
- проблемы preprocessing или shape mismatch, surfaced как user-visible validation/runtime error.

Backend semantics:

- retry не делать;
- логировать причину;
- переходить в безопасный fallback.

### `503 Service Unavailable`

Типичные причины:

- artifacts не найдены;
- service стартовал, но модель не загружена.

Backend semantics:

- допустим ограниченный повтор позже;
- response нельзя использовать как торговый сигнал;
- безопасный fallback обязателен.

### `500 Internal Server Error`

Типичные причины:

- сбой preprocessing;
- ошибка модели;
- неожиданный runtime failure.

Backend semantics:

- retry допустим ограниченно;
- сохранять причину ошибки;
- fallback в безопасное состояние.

## Retry And Fallback Expectations

Текущее backend behavior по коду:

- preflight health check возможен и по умолчанию обязателен через `strict_health_check: true`;
- retry выполняется только на network/url errors и HTTP `5xx`;
- retry count по умолчанию: `1`;
- на HTTP `4xx` retry не выполняется;
- при недоступности ML decision path уходит в `HOLD` fallback.

## Raw Data Contract

Raw candle contract задается кодом, а не JSON Schema.

Фактическая storage schema:

- `ticker`
- `timeframe`
- `begin`
- `end`
- `open`
- `high`
- `low`
- `close`
- `volume`
- `value`
- `source`

Storage format:

- Parquet

Backend-side quality checks:

- positive OHLC;
- `high >= low`;
- `volume > 0`;
- no duplicate `begin`;
- no mixed ticker/timeframe;
- sorted by `begin`;
- no overlapping intervals when `end` is known.

ML-side loader/cleaner then assumes that this storage remains compatible with `load.py` and `clean.py`.

## Compatibility / Change Policy

### Safe Changes

- добавить optional field в `POST /predict` response;
- добавить optional field в request;
- расширить `diagnostics`;
- документировать дополнительные assumptions без изменения shape.

### Breaking Changes

- изменить required fields;
- изменить endpoint path;
- изменить enum или semantics поля `action`;
- изменить naming convention полей;
- изменить minimum candle logic без синхронного изменения backend config и artifacts.

Если такие изменения понадобятся, безопаснее вводить новую версию контракта, а не silently менять текущий.
