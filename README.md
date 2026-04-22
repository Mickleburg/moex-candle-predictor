# MOEX Candle Predictor

`moex-candle-predictor` сейчас представляет собой связку из двух реализованных runtime-компонентов:

- `backend/` — Go HTTP backend для загрузки/сохранения свечей и вызова ML inference по HTTP.
- `ml/` — Python training pipeline и FastAPI inference service.

Также в репозитории есть:

- `shared/schemas/` — JSON Schema для контракта `POST /predict`;
- `data/` — директории для raw/processed/predictions/reports;
- `docs/` — техническая документация по текущему scope.

При расхождениях источники истины такие:

1. код;
2. checked-in runtime contracts и `shared/schemas/`;
3. приложенные PDF только как future reference.

## Что реализовано сейчас

- Go backend с HTTP API и entrypoint `backend/cmd/app/main.go`.
- Загрузка конфигурации из YAML с env overrides.
- Получение свечей из MOEX ISS / ALGOPACK через backend endpoints.
- Сохранение raw candles в `data/raw/*.parquet`.
- ML inference service c endpoints `GET /health` и `POST /predict`.
- Python training pipeline с загрузкой raw data, очисткой, time split, feature engineering, токенизацией, обучением и сохранением артефактов в `ml/artifacts/`.
- JSON Schema для request/response контракта `POST /predict`.
- Decision log в `data/reports/decision_log.jsonl`.

## Что не следует считать реализованным сейчас

- LLM/hybrid architecture из PDF не является текущей подтвержденной runtime-архитектурой.
- Production readiness не подтверждена.
- Свежий end-to-end запуск backend + ML в этой среде не подтвержден этой сверкой.
- Reproducible dataset в репозитории не зафиксирован: `data/raw/` сейчас пуст, кроме `.gitkeep`.
- Checked-in ML artifacts не следует считать гарантированно консистентными с текущим кодом feature pipeline.
- Автоматический полный backtest/evaluation pipeline не подключен к основному training entrypoint.

## High-Level Structure

```text
.
├── backend/            Go backend
├── data/               runtime data directories
├── docs/               technical documentation
├── ml/                 Python training + inference
└── shared/schemas/     JSON Schema for /predict
```

Подробная карта файлов: `docs/repository_map.md`.

## Current Runtime Scope

Текущий документируемый scope ограничен следующими частями:

- `backend/`
- `ml/`
- `shared/schemas/`
- `data/`

Критичные entrypoints:

- `backend/cmd/app/main.go`
- `backend/internal/app/app.go`
- `ml/src/models/train.py`
- `ml/src/service/api.py`

Критичный межсервисный контракт:

- backend -> ML: `GET /health`
- backend -> ML: `POST /predict`

## Minimal Local Run

Ниже приведены intended commands для локального запуска. Они описывают текущий код, но не подтверждают, что запуск был заново успешно воспроизведен в этой среде.

### 1. Поднять ML service

Пример для Windows PowerShell:

```powershell
Set-Location C:\Users\ancha\Projects\MOEX\moex-candle-predictor\ml
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn src.service.api:app --host 127.0.0.1 --port 8001
```

Если `py` недоступен, используйте явный путь к вашему Python 3.11+.

Health check:

```powershell
Invoke-RestMethod http://127.0.0.1:8001/health
```

### 2. Поднять backend

```powershell
Set-Location C:\Users\ancha\Projects\MOEX\moex-candle-predictor\backend
go run ./cmd/app -config config/config.yaml
```

Health check:

```powershell
Invoke-RestMethod http://127.0.0.1:8080/health
```

### 3. Минимальный backend -> ML path

После старта обоих сервисов backend использует:

- `ml.base_url: http://localhost:8001`
- `ml.health_path: /health`
- `ml.predict_path: /predict`

Подробный runtime flow: `docs/runtime_flow.md`.

## Expected Data And Artifacts

### Raw data

Training pipeline ожидает raw candles в `data/raw/` в формате, совместимом с:

- backend storage writer;
- `ml/src/data/load.py`;
- `ml/src/data/clean.py`.

Фактический raw contract:

- формат хранения: Parquet;
- базовые колонки: `ticker`, `timeframe`, `begin`, `end`, `open`, `high`, `low`, `close`, `volume`, `value`, `source`;
- данные должны относиться к одному ticker/timeframe на один training run или inference request;
- свечи должны быть хронологически отсортированы.

### ML artifacts

Inference service загружает артефакты из `ml/artifacts/`:

- `model.pkl`
- `tokenizer.pkl`
- `metadata.json`

Они позволяют сервису стартовать, но не являются доказательством train/serve consistency.

## Configs, Schemas, Docs

- backend config: `backend/config/config.yaml`
- ML configs: `ml/configs/data.yaml`, `ml/configs/features.yaml`, `ml/configs/train.yaml`, `ml/configs/eval.yaml`
- JSON Schema: `shared/schemas/candles_request.json`, `shared/schemas/prediction_response.json`
- architecture doc: `docs/architecture.md`
- API contract: `docs/api_contract.md`
- runtime flow: `docs/runtime_flow.md`
- known limits: `docs/known_limits.md`
- repository map: `docs/repository_map.md`
- experiments/evaluation scope: `docs/experiments.md`

## Current Status And Limits

- Эта сверка подтверждает структуру кода, entrypoints, контракты и checked-in artifacts на уровне code review.
- Эта сверка не подтверждает свежий успешный запуск backend, ML service или их end-to-end интеграции.
- `ml/.venv` закоммичен в репозиторий и не должен считаться переносимым или воспроизводимым окружением.
- `data/raw/` сейчас не содержит reproducible dataset.
- В `data/reports/decision_log.jsonl` есть реальные исторические записи, включая следы прошлых ошибок совместимости feature space.
- Leakage-related исправления в pipeline уже вносились, но checked-in artifacts не следует считать автоматически переобученными после этих изменений.

## Planned Future Architecture

В репозитории и в приложенном PDF описано направление развития в сторону гибридного trading-agent слоя:

- LLM technical-analysis agent;
- late-fusion aggregator;
- memory/orchestration;
- расширенная risk semantics.

Эту архитектуру нужно трактовать как planned/future extension. Даже если в `backend/` уже есть заготовки под такой путь, их нельзя считать подтвержденной основной runtime-архитектурой текущего проекта.
