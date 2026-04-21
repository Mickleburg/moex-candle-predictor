# MOEX Hybrid Trading Agent

Гибридный торговый агент для MOEX, в котором:

- алгоритмическая модель предсказывает следующие свечи и выдаёт числовой `algo signal`;
- LLM отдельно анализирует рынок, индикаторы и новости и выдаёт `llm signal` с risk flags;
- финальное решение `BUY / SELL / HOLD` принимает backend-агрегатор по жёстким правилам.

Это не “только ML-репозиторий”. Сейчас в проекте уже есть и ML pipeline, и рабочий Go backend, и end-to-end контур между ними.

## Что есть в проекте

### Backend

`backend/` отвечает за runtime-оркестрацию:

- получение рыночных данных из MOEX ISS;
- сохранение raw свечей в `Parquet`;
- вызов ML inference сервиса;
- отдельный LLM analysis слой;
- агрегацию `algo + llm + risk`;
- fallback-логику;
- логирование решений;
- API для внешнего использования.

### ML

`ml/` отвечает за модель и inference:

- загрузку и подготовку данных;
- feature engineering;
- токенизацию свечей;
- обучение baseline и LightGBM моделей;
- сохранение артефактов;
- FastAPI сервис `/health` и `/predict`.

### Данные

`data/` используется как общее хранилище:

- `data/raw/` — raw свечи;
- `data/processed/` — подготовленные датасеты;
- `data/predictions/` — предсказания/онлайн-артефакты;
- `data/reports/` — decision logs и отчёты.

## Текущая архитектура

```text
MOEX ISS / AlgoPack -> Backend (Go) -> ML service (FastAPI)
                               \-> LLM analyzer
                               \-> Risk manager
                               \-> Aggregator -> BUY / SELL / HOLD
```

Логика контура:

1. Backend получает рыночные данные.
2. Свечи сохраняются в `data/raw`.
3. Backend отправляет окно свечей в ML `/predict`.
4. Backend отдельно получает `llm signal`.
5. Агрегатор считает итоговый score.
6. Risk layer может понизить решение до `HOLD`.
7. Итоговое решение логируется.

## Что уже работает

### End-to-end MVP

На текущий момент уже реализовано:

- Go backend с конфигом и HTTP API;
- ML inference service на FastAPI;
- интеграция backend -> ML `/health` и `/predict`;
- сохранение raw свечей в `Parquet`;
- decision log в `JSONL`;
- гибридный late-fusion агрегатор;
- risk manager;
- LLM heuristic fallback;
- MOEX ISS integration;
- ALGOPACK-ready слой в backend;
- end-to-end вызов `POST /api/v1/decisions/evaluate`.

### Backend API

Уже есть:

- `GET /health`
- `POST /api/v1/candles/store`
- `GET /api/v1/candles/fetch`
- `POST /api/v1/decisions/evaluate`

MOEX / ISS:

- `GET /api/v1/moex/iss`
- `GET /api/v1/moex/security`
- `GET /api/v1/moex/candles`
- `GET /api/v1/moex/orderbook`
- `GET /api/v1/moex/trades`
- `GET /api/v1/moex/sitenews`

ALGOPACK:

- `GET /api/v1/algopack/dataset`
- `GET /api/v1/algopack/tradestats`
- `GET /api/v1/algopack/orderstats`
- `GET /api/v1/algopack/obstats`
- `GET /api/v1/algopack/hi2`
- `GET /api/v1/algopack/futoi`
- `GET /api/v1/algopack/realtime/candles`
- `GET /api/v1/algopack/realtime/orderbook`
- `GET /api/v1/algopack/realtime/trades`

### ML часть

Уже есть:

- загрузка данных из `Parquet`;
- валидация и подготовка свечей;
- feature engineering;
- tokenization;
- baseline модели;
- LightGBM pipeline;
- сохранение артефактов;
- FastAPI inference runtime.

## Что пока не готово

- модель обучена на mock-данных, а не на боевой истории;
- внешний production LLM/provider новостей ещё не подключён;
- веса агрегатора и пороги не откалиброваны бэктестом;
- нет paper trading;
- нет execution слоя;
- нет production scheduler/monitoring/alerting;
- `Super Candles`, `OBStats`, `OrderStats`, `TradeStats`, `HI2` зависят от тарифа ALGOPACK.

## Текущий статус по AlgoPack

Backend уже умеет работать и с обычным ISS, и с ALGOPACK.

Разделение источников:

- `iss.moex.com` — обычный ISS;
- `apim.moex.com` — ALGOPACK / datashop.

Сейчас в проекте предусмотрен `Bearer APIKEY` flow для ALGOPACK, но фактически доступные датасеты зависят от тарифа MOEX.

На текущем `Стартовом` пакете по скринам доступны:

- 15-минутно задержанные свечи и сделки;
- `FUTOI`;
- API;
- Python библиотека.

Пока недоступны на текущем тарифе:

- `OBStats`;
- `OrderStats`;
- `TradeStats`;
- `HI2`;
- `Super Candles`.

То есть для MVP сейчас разумно опираться на:

- свечи;
- сделки;
- опционально `FUTOI`.

## Структура репозитория

```text
moex-candle-predictor/
├── backend/              # Go backend
├── data/                 # Общее хранилище данных
│   ├── raw/
│   ├── processed/
│   ├── predictions/
│   └── reports/
├── docs/                 # Архитектура и договорённости
├── ml/                   # ML pipeline и inference service
├── shared/               # Общие схемы/контракты
├── .gitignore
└── README.md
```

## Быстрый старт

### 1. Запуск ML

```bash
cd /Users/main/Desktop/moex-candle-predictor/ml
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn src.service.api:app --host 127.0.0.1 --port 8001
```

Проверка:

```bash
curl http://127.0.0.1:8001/health
```

### 2. Запуск backend

```bash
cd /Users/main/Desktop/moex-candle-predictor/backend
GOCACHE=/tmp/go-build go run ./cmd/app -config config/config.yaml
```

Проверка:

```bash
curl http://127.0.0.1:8080/health
```

### 3. Проверка гибридного решения

Пример end-to-end вызова:

```bash
python3 - <<'PY' | curl -s http://127.0.0.1:8080/api/v1/decisions/evaluate -H 'Content-Type: application/json' -d @-
from datetime import datetime, timedelta
import json, sys

base = datetime(2024, 1, 1, 10, 0, 0)
price = 250.0
candles = []

for i in range(40):
    o = price
    c = price + 0.2
    candles.append({
        "begin": (base + timedelta(hours=i)).isoformat(),
        "open": round(o, 2),
        "high": round(max(o, c) + 0.3, 2),
        "low": round(min(o, c) - 0.3, 2),
        "close": round(c, 2),
        "volume": 100000 + i * 1000,
        "ticker": "SBER",
        "timeframe": "1H"
    })
    price = c

payload = {
    "candles": candles,
    "indicators": {
        "rsi": 58.0,
        "macd_line": 1.2,
        "macd_signal": 0.7,
        "ema20": 252.4,
        "ema50": 249.8,
        "volume_ratio": 1.3,
        "support_levels": [249.0, 247.5],
        "resistance_levels": [253.0, 255.0]
    },
    "portfolio": {
        "current_position": 0.0,
        "exposure": 0.1,
        "day_pnl": 0.0,
        "equity": 1000000.0
    }
}

json.dump(payload, sys.stdout)
PY
```

## Контракт между backend и ML

Backend ожидает от ML:

- стабильный `/health`;
- стабильный `/predict`;
- понятную семантику `predicted_token`, `probabilities`, `confidence`;
- консистентный `model_version`;
- согласованный `horizon`.

ML получает от backend:

- последовательность свечей одного тикера и одного таймфрейма;
- свечи отсортированы по времени от старых к новым;
- raw схема совместима с `OHLCV + value`.

## Кто за что отвечает

### Backend

- интеграция с MOEX / ISS / ALGOPACK;
- raw storage;
- API;
- orchestration runtime;
- вызов ML и LLM;
- агрегатор;
- risk layer;
- decision logging.

### ML

- подготовка признаков;
- training pipeline;
- inference logic;
- артефакты модели;
- качество модели;
- backtest / calibration.

## Документация

Если нужна детализация по отдельным частям:

- `backend/README.md` — backend runtime, API, ISS и ALGOPACK integration;
- `ml/README.md` — ML pipeline и inference service;
- `docs/architecture.md` — общая архитектура;
- `docs/api_contract.md` — контракт между сервисами;
- `docs/experiments.md` — ML эксперименты и оценка.

## Текущий практический вывод

Проект уже годится для:

- демонстрации общей архитектуры;
- интеграции backend и ML;
- сбора и хранения рыночных данных;
- запуска гибридного decision pipeline.

Проект пока не годится для:

- реальной торговли;
- production risk/execution;
- оценки качества модели на реальной истории без переобучения и калибровки.

## Следующий шаг

Самый полезный следующий шаг для команды сейчас:

1. Собирать реальные свечи MOEX в `data/raw`.
2. Переобучить ML на реальных данных.
3. Зафиксировать финальный контракт `AlgoSignal`.
4. Подобрать веса агрегатора и пороги.
5. Только потом двигаться в paper trading.
