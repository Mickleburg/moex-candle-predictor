# Backend Documentation

## Что это

`backend/` — это Go-часть проекта гибридного торгового агента.

Она отвечает не за обучение модели, а за оркестрацию:

- получение свечей;
- сохранение raw-данных;
- вызов ML inference сервиса;
- получение отдельного LLM-сигнала;
- агрегацию сигналов по жёстким правилам;
- risk-check;
- логирование итогового решения.

Идея системы такая:

1. Алгоритмический candle predictor даёт числовой сигнал.
2. LLM-модуль отдельно даёт качественную интерпретацию рынка и risk flags.
3. Финальное решение `BUY / SELL / HOLD` принимает не LLM, а backend-агрегатор.

## Что уже сделано

На текущий момент реализован рабочий MVP-контур.

### Backend

Готово:

- поднят Go backend;
- есть загрузка конфига из YAML;
- есть health endpoint;
- есть endpoint для оценки финального торгового решения;
- есть endpoint для fetch свечей из MOEX;
- есть endpoint для сохранения свечей в raw storage;
- есть хранение raw-свечей в Parquet;
- есть логирование решений в JSONL;
- есть risk manager;
- есть LLM analyzer как отдельный слой;
- есть агрегатор сигналов;
- есть safe fallback, если ML недоступен.

### ML integration

Готово:

- backend умеет ходить в ML `GET /health`;
- backend умеет ходить в ML `POST /predict`;
- backend умеет использовать ответ ML как `AlgoSignal`;
- backend корректно переживает ошибки ML и переходит в `HOLD`.

### End-to-end

Проверено:

- ML service поднимается;
- backend поднимается;
- `GET /health` на ML работает;
- `GET /health` на backend работает;
- `POST /api/v1/decisions/evaluate` проходит end-to-end;
- финальное решение считается backend-агрегатором, а не fallback-веткой.

## Что мы починили по пути

Во время сборки и запуска были исправлены следующие проблемы:

- сломанные импорты в ML runtime;
- лишняя зависимость inference от training-модуля;
- несовместимость имён функций `save_json` / `write_json`;
- неверные пути к `data/raw` и `ml/artifacts`;
- пустые артефакты модели;
- несовместимость train-конфига и `LGBMClassifier`;
- конфликт параметров LightGBM;
- несовместимость форматов времени между Python и Go;
- несовпадение схемы inference из-за отсутствующего поля `value` в runtime-запросе;
- загрузка старых процессов вместо новых при перезапуске сервисов.

## Что пока не готово

Есть важные ограничения MVP.

- Модель сейчас обучена на mock-данных, а не на реальных свечах MOEX.
- Настоящий внешний LLM или news-провайдер пока не подключён.
- Сейчас используется `heuristic-fallback` для LLM-слоя.
- Веса и пороги агрегатора ещё не откалиброваны по бэктесту.
- Нет production scheduler-а для регулярного hourly запуска.
- Нет мониторинга и алёртинга.
- Нет paper trading / execution слоя.
- Нет боевой торговой логики с ордерами.

## Кто за что отвечает

### Зона ответственности ML

ML-команда отвечает за:

- реальные данные для обучения в `data/raw`;
- корректный training pipeline;
- рабочие артефакты:
  - `ml/artifacts/model.pkl`
  - `ml/artifacts/tokenizer.pkl`
  - `ml/artifacts/metadata.json`
- качество модели;
- калибровку confidence;
- подбор порогов и весов по истории;
- согласованный контракт `horizon`, `direction`, `confidence`.

### Зона ответственности backend

Backend-команда отвечает за:

- получение свечей из MOEX;
- сохранение raw-свечей в правильной схеме;
- вызов ML `/health` и `/predict`;
- вызов LLM analyzer;
- агрегацию сигналов;
- risk manager;
- fallback-логику;
- API для внешнего использования;
- логирование решений;
- оркестрацию полного runtime-контура.

### Что общее

Общий контракт между backend и ML:

- структура входных свечей;
- структура ответа ML;
- интерпретация `predicted_token`, `action`, `confidence`;
- семантика `horizon`;
- правила совместимости схем.

## Текущая архитектура backend

Основные части:

- `cmd/app/main.go` — вход в backend приложение;
- `internal/config` — загрузка и валидация конфигурации;
- `internal/app` — wiring HTTP server и handlers;
- `internal/moex` — клиент MOEX ISS;
- `internal/service/history_service.go` — валидация и сохранение свечей;
- `internal/service/predict_service.go` — интеграция с ML;
- `internal/service/llm_service.go` — LLM analyzer;
- `internal/service/decision_service.go` — orchestration полного решения;
- `internal/service/risk_service.go` — risk checks;
- `internal/storage/files.go` — Parquet raw storage и decision logs;
- `internal/models` — общие доменные структуры.

## Endpoint-ы backend

### `GET /health`

Проверяет:

- backend runtime;
- доступность ML;
- состояние LLM analyzer.

Пример:

```bash
curl http://127.0.0.1:8080/health
```

### `POST /api/v1/candles/store`

Сохраняет raw-свечи в Parquet.

### `GET /api/v1/candles/fetch`

Забирает свечи из MOEX и сохраняет их в raw storage.

Пример:

```bash
curl "http://127.0.0.1:8080/api/v1/candles/fetch?ticker=SBER&timeframe=1H&from=2024-01-01&to=2024-01-10"
```

### `POST /api/v1/decisions/evaluate`

Главный endpoint гибридного агента.

На вход получает:

- последние свечи;
- индикаторы;
- optional news;
- состояние портфеля.

На выходе возвращает:

- `algo_signal`;
- `llm_signal`;
- `aggregation`;
- `risk`;
- итоговый `action`.

## Endpoint-ы ML

### `GET /health`

Пример:

```bash
curl http://127.0.0.1:8001/health
```

### `POST /predict`

На вход:

- минимум 32 свечи;
- свечи должны идти по времени от старых к новым;
- все свечи должны быть одного тикера и одного таймфрейма.

## Конфигурация

Основной backend config:

- `backend/config/config.yaml`

В нём задаются:

- адрес backend;
- настройки MOEX;
- адрес ML сервиса;
- настройки LLM analyzer;
- веса агрегатора;
- риск-лимиты;
- пути к raw storage и decision log.

## Формат raw storage

Raw свечи сохраняются в Parquet.

Используемая схема:

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

Это соответствует ожиданиям ML pipeline.

## Логика принятия решения

### 1. AlgoSignal

Backend получает из ML:

- `predicted_token`
- `probabilities`
- `action`
- `confidence`
- `model_version`

Из этого строится нормализованный `AlgoSignal`.

### 2. LLMSignal

LLM-модуль возвращает:

- `direction`
- `strength`
- `confidence`
- `key_factors`
- `risk_flags`
- `horizon`

Сейчас по умолчанию работает `heuristic-fallback`.

### 3. Агрегация

Используется late fusion:

```text
final_score = w_algo * algo.direction * algo.confidence
            + w_llm  * llm.direction  * llm.strength * llm.confidence
```

Стартовые веса:

- `w_algo = 0.6`
- `w_llm = 0.4`

Стартовые пороги:

- `BUY`, если score >= `0.35`
- `SELL`, если score <= `-0.35`
- иначе `HOLD`

Если у LLM есть `risk_flags`, backend может принудительно сделать `HOLD`.

### 4. Risk Manager

После агрегации применяется risk layer:

- лимит позиции;
- лимит exposure;
- дневной loss limit;
- запрет short, если он выключен;
- расчёт допустимого размера позиции.

## Что уже проверено на практике

Были успешно пройдены следующие проверки:

- `go build ./...`
- `go test ./...`
- `ml/test_smoke.py`
- запуск ML service;
- запуск backend;
- успешный `GET /health` на обоих сервисах;
- успешный запрос в `POST /api/v1/decisions/evaluate`.

## Как запускать

### 1. Запуск ML

Из каталога `ml/`:

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

Ожидается:

- `status = healthy`
- `model_loaded = true`

### 2. Запуск backend

Из каталога `backend/`:

```bash
cd /Users/main/Desktop/moex-candle-predictor/backend
GOCACHE=/tmp/go-build go run ./cmd/app -config config/config.yaml
```

Проверка:

```bash
curl http://127.0.0.1:8080/health
```

Ожидается:

- backend healthy;
- ML healthy;
- LLM healthy.

## Как тестировать систему

### Smoke-test ML

```bash
cd /Users/main/Desktop/moex-candle-predictor/ml
source .venv/bin/activate
python test_smoke.py
```

### Smoke-test backend

```bash
cd /Users/main/Desktop/moex-candle-predictor/backend
GOCACHE=/tmp/go-build go test ./...
```

### Полный end-to-end test

Запрос в гибридный агент:

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

Что должно быть в ответе:

- не должно быть `ml_unavailable_fallback`;
- `algo_signal` должен быть заполнен;
- `llm_signal` должен быть заполнен;
- `aggregation` должна быть заполнена;
- итоговый `action` должен считаться агрегатором;
- `risk` должен быть валидным объектом.

## Как интерпретировать успешный ответ

Примерно такой ответ означает, что система работает корректно:

- `algo_signal` пришёл из ML;
- `llm_signal` пришёл из отдельного LLM слоя;
- `score` посчитан агрегатором;
- итоговое действие зависит от порогов;
- если `score` внутри hold-band, будет `HOLD`.

Например:

- слабый algo-сигнал;
- умеренно bullish LLM;
- score меньше buy threshold;
- итоговое решение: `HOLD`.

Это нормальное поведение.

## Важное замечание

Сейчас end-to-end работает технически, но не является торгово готовым.

Почему:

- модель обучена на mock-свечах;
- LLM analyzer пока fallback;
- нет калибровки по реальной истории;
- нет paper trading цикла;
- нет execution слоя.

То есть на данном этапе система уже готова для демонстрации backend architecture и интеграции,
но не для реальной торговли.

## Следующий правильный шаг

### Backend делает

- стабильный сбор реальных свечей MOEX;
- запись их в `data/raw`;
- регулярный runtime-контур;
- логи и monitoring hooks;
- подготовку к paper trading.

### ML делает

- обучение на реальных свечах;
- выдачу боевых артефактов;
- калибровку confidence и thresholds;
- backtest / walk-forward validation.

### Вместе

- фиксируем контракт сигналов;
- унифицируем `horizon`;
- подбираем веса агрегатора;
- готовим демонстрационный сценарий для проекта.

## Где смотреть код

- [main.go](/Users/main/Desktop/moex-candle-predictor/backend/cmd/app/main.go)
- [config.yaml](/Users/main/Desktop/moex-candle-predictor/backend/config/config.yaml)
- [app.go](/Users/main/Desktop/moex-candle-predictor/backend/internal/app/app.go)
- [config.go](/Users/main/Desktop/moex-candle-predictor/backend/internal/config/config.go)
- [predict_service.go](/Users/main/Desktop/moex-candle-predictor/backend/internal/service/predict_service.go)
- [llm_service.go](/Users/main/Desktop/moex-candle-predictor/backend/internal/service/llm_service.go)
- [decision_service.go](/Users/main/Desktop/moex-candle-predictor/backend/internal/service/decision_service.go)
- [risk_service.go](/Users/main/Desktop/moex-candle-predictor/backend/internal/service/risk_service.go)
- [history_service.go](/Users/main/Desktop/moex-candle-predictor/backend/internal/service/history_service.go)
- [files.go](/Users/main/Desktop/moex-candle-predictor/backend/internal/storage/files.go)

