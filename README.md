# MOEX Candle Predictor

Репозиторий состоит из двух основных частей:

- `backend/` — Go-сервис для оркестрации: получение свечей из MOEX, вызов ML inference, эвристический LLM-layer, агрегация сигнала, риск-проверка, логирование решений.
- `ml/` — Python pipeline для подготовки данных, обучения модели и FastAPI inference-сервиса.

Дополнительно:

- `data/` — локальное хранилище raw/processed/predictions/reports.
- `shared/schemas/` — JSON Schema для ML request/response.
- `docs/` — архитектура, API-контракт и заметки по обучению.

## Что реально реализовано

- В backend есть HTTP API, MOEX ISS/ALGOPACK proxy endpoints, сохранение raw свечей в Parquet и decision log в JSONL.
- В ML есть training pipeline, артефакты модели, FastAPI endpoints `/health` и `/predict`.
- Между backend и ML есть рабочий HTTP-контракт на уровне полей запроса и ответа.

## Что важно знать перед запуском

- Репозиторий не содержит reproducible dev environment: в нём нет зафиксированного Go toolchain setup, а `ml/.venv` закоммичен в репозиторий, но не должен считаться переносимым окружением.
- Закоммиченные `ml/artifacts/*` позволяют поднять inference, но сами по себе не гарантируют корректность модели или совместимость после изменения feature pipeline.
- В проекте есть существенные архитектурные риски в train/serve части ML. Они описаны в аудите и не должны игнорироваться при handoff.

## Быстрый запуск

### ML

```bash
cd ml
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
uvicorn src.service.api:app --host 127.0.0.1 --port 8001
```

### Backend

```bash
cd backend
go run ./cmd/app -config config/config.yaml
```

### Smoke checks

```bash
curl http://127.0.0.1:8001/health
curl http://127.0.0.1:8080/health
```

## Основные точки входа

- `backend/cmd/app/main.go`
- `backend/internal/app/app.go`
- `ml/src/models/train.py`
- `ml/src/service/api.py`

## Документация

- `backend/README.md`
- `ml/README.md`
- `docs/architecture.md`
- `docs/api_contract.md`
- `docs/experiments.md`
