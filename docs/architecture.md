# Архитектура

## Компоненты

### Backend

Go-сервис из `backend/`:

- читает конфиг;
- поднимает HTTP API;
- получает свечи из MOEX ISS / ALGOPACK;
- сохраняет raw candles в `data/raw`;
- вызывает ML service;
- строит итоговое решение через aggregation + risk.

### ML

Python-код из `ml/`:

- загружает raw candles из `data/raw`;
- готовит признаки и токены;
- обучает модель;
- сохраняет `model.pkl`, `tokenizer.pkl`, `metadata.json`;
- поднимает FastAPI inference.

### Shared

`shared/schemas/` содержит JSON Schema для request/response контракта ML API.

### Data

- `data/raw/` — Parquet со свечами;
- `data/predictions/` — online prediction artifacts;
- `data/reports/` — decision log и отчёты;
- `data/processed/` — зарезервировано под промежуточные данные.

## Runtime flow

1. Backend получает свечи из MOEX или принимает их через API.
2. Свечи валидируются и при необходимости сохраняются в `data/raw/`.
3. Backend отправляет окно свечей в ML `/predict`.
4. Backend получает отдельный LLM signal или fallback.
5. Aggregator считает итоговый score.
6. Risk layer может заблокировать `BUY`/`SELL` и превратить решение в `HOLD`.
7. Decision response логируется в JSONL.

## Training flow

1. `ml/src/data/load.py` читает raw data.
2. `ml/src/data/clean.py` чистит и сортирует данные.
3. `ml/src/data/split.py` делает time split.
4. `ml/src/features/indicators.py` считает признаки.
5. `ml/src/features/tokenizer.py` строит токены.
6. `ml/src/features/windows.py` формирует training windows.
7. `ml/src/models/train.py` обучает модель и пишет артефакты.

## Критичные стыки

- backend -> ml request/response contract;
- backend storage schema -> ml training loader;
- configs -> runtime paths;
- `ml/artifacts/*` -> inference service.

## Известные риски

- Train и serve логика ML не эквивалентны по семантике токенов; это нужно считать критическим риском, а не допустимым MVP-компромиссом.
- Документация и конфиги содержат параметры, которые не всегда реально используются кодом.
- Репозиторий хранит локальные runtime artifacts (`ml/.venv`, `ml/artifacts`) вместе с исходниками, что усложняет handoff и воспроизводимость.
