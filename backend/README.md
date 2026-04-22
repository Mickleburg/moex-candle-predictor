# Backend

`backend/` — Go HTTP backend проекта. Его текущая подтверждаемая роль в системе: принять или загрузить свечи, привести их к runtime-ограничениям backend, сохранить raw data в `data/raw/` и вызвать ML inference service по HTTP.

## Назначение

Backend отвечает за orchestration вокруг рыночных данных и ML inference:

- загрузка конфигурации;
- подъем HTTP API;
- получение данных из MOEX ISS / ALGOPACK;
- прием свечей от клиента;
- валидация и нормализация raw candle batches;
- запись raw candles в Parquet;
- health check ML-сервиса;
- вызов `POST /predict`;
- обработка ошибок и fallback до безопасного `HOLD`.

## Зона ответственности

### Backend должен делать

- работать только с raw OHLCV-данными и transport-level контрактами;
- обеспечивать один ticker и один timeframe в одном candle batch;
- сортировать свечи по `begin`;
- не сохранять очевидно битые свечи;
- ограниченно retry-ить ML только на сетевых/5xx ошибках;
- логировать решения в `data/reports/decision_log.jsonl`.

### Backend не должен делать

- считать индикаторы для ML;
- считать токены;
- повторять Python preprocessing;
- импортировать ML-логику напрямую;
- менять семантику `action`, возвращаемую ML;
- считать checked-in ML artifacts заведомо валидными без отдельной проверки.

## Runtime Flow

Основной подтверждаемый flow выглядит так:

1. Приложение стартует из `cmd/app/main.go`.
2. `internal/config` загружает `config.yaml`, env overrides и резолвит относительные storage paths.
3. `internal/app/app.go` собирает `HistoryService`, `PredictService`, storage и MOEX client.
4. HTTP handlers принимают candles или запрашивают их из MOEX.
5. `HistoryService.PrepareCandles` валидирует batch и ставит defaults.
6. `FileStore.SaveRawCandles` пишет Parquet в `data/raw/`.
7. `PredictService.Health` и `PredictService.PredictAlgoSignal` обращаются к ML service.
8. Ошибки ML переводятся в безопасный fallback на стороне decision flow.

Отдельно от этого в backend есть расширенный decision path, но его нужно трактовать осторожно: он тянется в сторону planned hybrid architecture и не должен подменять собой базовый backend<->ML contract.

## Key Packages And Files

- `cmd/app/main.go` — CLI entrypoint, загрузка конфига и lifecycle приложения.
- `internal/app/app.go` — wiring зависимостей, HTTP routes, handlers.
- `internal/config/config.go` — структура конфига, defaults, env overrides, path resolution, validation.
- `internal/models/candle.go` — runtime candle model, JSON parsing, defaults, timeframe parsing.
- `internal/models/prediction.go` — модели ML request/response и decision-related structs.
- `internal/service/history_service.go` — валидация batch и сохранение raw candles.
- `internal/service/predict_service.go` — HTTP client к ML service, retry policy, parsing `/health` и `/predict`.
- `internal/storage/files.go` — Parquet storage для raw candles и JSONL decision log.
- `internal/moex/client.go` — HTTP client к MOEX ISS / ALGOPACK.
- `config/config.yaml` — текущий backend runtime config.

## Работа С Raw Data

Backend хранит raw candles как Parquet. Формат файла фактически задается `internal/storage/files.go`.

Столбцы Parquet row:

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

`HistoryService.PrepareCandles` применяет следующие ограничения:

- batch не должен быть пустым;
- `begin` обязателен;
- `open/high/low/close` должны быть положительными;
- `high >= low`;
- `volume > 0`;
- свечи сортируются по `begin`;
- нельзя смешивать несколько ticker;
- нельзя смешивать несколько timeframe;
- нельзя дублировать `begin`;
- нельзя иметь overlapping intervals, если у предыдущей свечи заполнен `end`.

Defaults backend:

- ticker: `market.default_ticker`
- timeframe: `market.default_timeframe`
- source: `market.default_source` или request source
- value: `close * volume`, если отсутствует
- end: вычисляется из timeframe, если отсутствует

## Вызов ML

Backend работает с ML service только по HTTP.

Подтвержденные endpoints:

- `GET /health`
- `POST /predict`

Текущие параметры по умолчанию в `config/config.yaml`:

- `ml.base_url: http://localhost:8001`
- `ml.min_candles: 32`
- `ml.strict_health_check: true`
- `ml.retry_count: 1`
- `ml.timeout_seconds: 5`

Поведение `PredictService`:

- не вызывает `POST /predict`, если свечей меньше `ml.min_candles`;
- при `strict_health_check: true` сначала требует healthy/model_loaded от ML;
- retry делает только на сетевых ошибках, timeout/url errors и HTTP `5xx`;
- на HTTP `4xx` retry не делает.

## HTTP Endpoints Backend

Реально зарегистрированные routes:

- `GET /health`
- `POST /api/v1/candles/store`
- `GET /api/v1/candles/fetch`
- `POST /api/v1/decisions/evaluate`
- `GET /api/v1/moex/iss`
- `GET /api/v1/moex/security`
- `GET /api/v1/moex/candles`
- `GET /api/v1/moex/orderbook`
- `GET /api/v1/moex/trades`
- `GET /api/v1/moex/sitenews`
- `GET /api/v1/algopack/dataset`
- `GET /api/v1/algopack/tradestats`
- `GET /api/v1/algopack/orderstats`
- `GET /api/v1/algopack/obstats`
- `GET /api/v1/algopack/hi2`
- `GET /api/v1/algopack/futoi`
- `GET /api/v1/algopack/realtime/candles`
- `GET /api/v1/algopack/realtime/orderbook`
- `GET /api/v1/algopack/realtime/trades`

Из них текущий ключевой интеграционный контракт с ML касается только backend health aggregation и вызова `POST /predict`.

## Known Limitations And Assumptions

- Основной persistence layer здесь файловый, а не полноценная БД.
- `GET /health` backend агрегирует статус backend + ML + LLM component, но это не равно end-to-end business validation.
- `storage.raw_dir` и `storage.decision_log_path` зависят от корректного path resolution из конфига.
- Raw candle contract формально не описан JSON Schema; он задается кодом backend storage и ML loader/cleaner.
- Default current assumptions: один ticker, один timeframe, минимум 32 свечи на inference request.
- Свежий успешный backend build/run в этой среде этой сверкой не подтвержден.

## Out Of Scope And Experimental Hybrid Pieces

В backend уже есть код, связанный с decision aggregation, LLM analyzer и risk logic:

- `internal/service/decision_service.go`
- `internal/service/llm_service.go`
- `internal/service/risk_service.go`
- endpoint `POST /api/v1/decisions/evaluate`

Этот код существует в репозитории, но его не следует описывать как подтвержденную базовую архитектуру текущего runtime scope. Он ближе к planned/future hybrid path:

- shared schemas для него отсутствуют;
- его поведение не зафиксировано отдельным стабильным контрактом;
- он опирается на heuristic fallback или внешний LLM provider;
- свежая runtime verification этого пути не подтверждена.

Поэтому при onboarding новый разработчик должен считать primary backend responsibility именно raw-data ingestion/storage и HTTP integration с ML inference service.
