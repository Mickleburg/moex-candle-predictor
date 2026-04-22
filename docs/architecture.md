# Architecture

## Source-Of-Truth Hierarchy

Для этой кодовой базы порядок источников истины должен быть таким:

1. код;
2. `shared/schemas/` и фактические runtime contracts;
3. PDF-документы как future reference.

Следствия:

- если README или docs расходятся с кодом, правдой считается код;
- если PDF описывает более широкую hybrid/LLM architecture, это не делает ее текущей реализацией;
- `shared/schemas/` важны только там, где они реально существуют, то есть для `POST /predict`.

## Current Architecture: Only Implemented System

Документ ниже описывает только текущий реализованный scope:

- `backend/`
- `ml/`
- `shared/schemas/`
- `data/`

### Backend

Подтверждаемые entrypoints:

- `backend/cmd/app/main.go`
- `backend/internal/app/app.go`

Подтверждаемая роль backend:

- поднять HTTP API;
- загрузить/принять свечи;
- провалидировать candle batch;
- сохранить raw candles в `data/raw/`;
- вызвать ML service по HTTP;
- обработать response и error semantics.

### ML

Подтверждаемые entrypoints:

- `ml/src/models/train.py`
- `ml/src/service/api.py`

Подтверждаемая роль ML:

- training pipeline от raw candles до artifacts;
- inference service с `GET /health` и `POST /predict`;
- preprocessing внутри Python-кода;
- загрузка `model.pkl`, `tokenizer.pkl`, `metadata.json`.

### Shared Schemas

`shared/schemas/` сейчас покрывает только contract для `POST /predict`:

- `shared/schemas/candles_request.json`
- `shared/schemas/prediction_response.json`

Эти схемы не описывают:

- raw data schema в `data/raw/`;
- backend health payload;
- behavior decision/LLM path.

### Data Layer

Текущие runtime directories:

- `data/raw/` — raw candles storage;
- `data/processed/` — зарезервировано под промежуточные данные;
- `data/predictions/` — зарезервировано под prediction outputs;
- `data/reports/` — decision log и возможные отчеты.

Фактическое состояние checked-in repo:

- `data/raw/` пуст, кроме `.gitkeep`;
- `data/reports/decision_log.jsonl` содержит исторические записи decision flow.

## Critical Contracts

### Raw Candle Contract

Raw candle contract задается не JSON Schema, а комбинацией:

- `backend/internal/storage/files.go`
- `backend/internal/service/history_service.go`
- `ml/src/data/load.py`
- `ml/src/data/clean.py`

Фактически ожидаются поля:

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

### Backend -> ML Contract

Подтвержденный межсервисный контракт:

- `GET /health`
- `POST /predict`

JSON Schema есть только для `POST /predict`. Дополнительные runtime assumptions фиксируются кодом:

- минимум `32` свечи;
- chronological ordering;
- один ticker;
- один timeframe.

## Current Data Flow

### Flow A: ingest/store raw candles

1. backend получает candles от клиента или MOEX;
2. `HistoryService.PrepareCandles` проверяет batch;
3. `FileStore.SaveRawCandles` пишет Parquet в `data/raw/`.

### Flow B: train pipeline

1. `ml/src/models/train.py` читает `data/raw/`;
2. loader/cleaner подготавливают DataFrame;
3. pipeline строит features/tokens/windows;
4. модель и metadata пишутся в `ml/artifacts/`.

### Flow C: inference

1. `ml/src/service/api.py` поднимает FastAPI app;
2. `CandlePredictor` загружает artifacts;
3. `/predict` строит features на входных candles и возвращает prediction payload.

### Flow D: backend -> ML request path

1. backend проверяет `/health`;
2. backend отправляет `POST /predict`;
3. backend парсит `PredictResponse`;
4. при ошибках использует fallback semantics.

## Planned / Future Architecture

Приложенный PDF про гибридного торгового агента нужно трактовать как planned extension, а не как текущую runtime-архитектуру.

К future/planned scope относятся:

- отдельный LLM technical-analysis agent;
- late-fusion aggregator как основная архитектурная ось;
- memory/orchestration layers;
- расширенный hybrid decision loop;
- richer risk semantics и prompt-governed structured LLM outputs.

В `backend/` уже есть кодовые заготовки, которые движутся в эту сторону:

- `internal/service/decision_service.go`
- `internal/service/llm_service.go`
- `internal/service/risk_service.go`

Но их нужно описывать как experimental/planned code present in repo, а не как подтвержденную текущую архитектуру системы.
