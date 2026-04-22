# Backend

`backend/` — Go runtime-слой проекта. Он не обучает модель; его задача — собрать рыночные данные, вызвать ML и вернуть итоговое решение.

## Ответственность

- загрузка конфигурации из YAML и env overrides;
- HTTP API;
- получение данных из MOEX ISS и ALGOPACK;
- сохранение raw свечей в Parquet;
- вызов ML `/health` и `/predict`;
- вызов LLM analyzer или эвристического fallback;
- агрегация `algo + llm`;
- risk-check;
- decision log в JSONL.

## Точки входа

- `cmd/app/main.go` — запуск приложения;
- `internal/app/app.go` — wiring HTTP handlers;
- `internal/moex/` — клиент ISS / ALGOPACK;
- `internal/service/` — orchestration, ML, LLM, risk, history;
- `internal/storage/files.go` — Parquet и JSONL storage.

## API

Реально реализованные endpoints:

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

## Конфигурация

Основной файл: `config/config.yaml`.

Критичные секции:

- `server` — адрес и timeouts;
- `market` — дефолтный ticker/timeframe;
- `moex` — ISS / ALGOPACK endpoints и auth;
- `ml` — базовый URL inference-сервиса;
- `llm` — внешний analyzer или heuristic fallback;
- `aggregator` — веса и пороги;
- `risk` — лимиты позиции и экспозиции;
- `storage` — пути до raw storage и decision log.

Есть env overrides для `PORT`, `BACKEND_ADDR`, `ML_BASE_URL`, `MOEX_*`, `LLM_BASE_URL`, `DEFAULT_TICKER`, `DEFAULT_TIMEFRAME`, `MAX_POSITION_ABS`.

## Запуск

```bash
cd backend
go run ./cmd/app -config config/config.yaml
```

## Ограничения

- В проекте нет полноценного persistence layer кроме файлового storage.
- Нет scheduler, monitoring, alerting и execution layer.
- Поведение `POST /api/v1/decisions/evaluate` зависит от доступности ML-сервиса и от текущих артефактов модели.
- Документация в этом каталоге не должна трактоваться как подтверждение production readiness.
