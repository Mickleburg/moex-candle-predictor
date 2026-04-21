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
- есть auth-aware клиент для MOEX ISS / AlgoPack;
- есть универсальный JSON proxy для `/iss/...` ресурсов;
- есть shortcut endpoint-ы для `security`, `candles`, `orderbook`, `trades`, `sitenews`;
- есть отдельный namespace `/api/v1/algopack/*` для уникальных датасетов ALGOPACK;
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

### `GET /api/v1/moex/iss`

Универсальный proxy к JSON-ресурсам ISS / AlgoPack.

Используется для любого ресурса `/iss/...`, который backend должен пробросить без написания отдельного handler-а.

Примеры:

```bash
curl "http://127.0.0.1:8080/api/v1/moex/iss?path=/iss/securities/SBER"
curl "http://127.0.0.1:8080/api/v1/moex/iss?path=/iss/engines/stock/markets/shares/boards/TQBR/securities/SBER/candles&from=2024-01-01&till=2024-01-10&interval=60&iss.meta=off&iss.only=candles"
```

Важно:

- `path` передаётся отдельно от query string;
- остальные query-параметры backend пробрасывает как есть;
- разрешены только ресурсы внутри `/iss/...`;
- proxy поддерживает только JSON-ответы.

### `GET /api/v1/moex/security`

Получить список бумаг или спецификацию одной бумаги.

Примеры:

```bash
curl "http://127.0.0.1:8080/api/v1/moex/security"
curl "http://127.0.0.1:8080/api/v1/moex/security?security=SBER"
```

### `GET /api/v1/moex/candles`

Получить raw JSON по свечам из ISS без сохранения в storage.

Пример:

```bash
curl "http://127.0.0.1:8080/api/v1/moex/candles?ticker=SBER&timeframe=1H&from=2024-01-01&to=2024-01-10"
```

### `GET /api/v1/moex/orderbook`

Получить стакан по инструменту.

Пример:

```bash
curl "http://127.0.0.1:8080/api/v1/moex/orderbook?security=SBER&depth=20"
```

### `GET /api/v1/moex/trades`

Получить сделки по инструменту.

Пример:

```bash
curl "http://127.0.0.1:8080/api/v1/moex/trades?security=SBER&limit=100"
```

### `GET /api/v1/moex/sitenews`

Получить биржевые новости из ISS.

Пример:

```bash
curl "http://127.0.0.1:8080/api/v1/moex/sitenews?limit=20"
```

## Endpoint-ы ALGOPACK

В backend теперь есть отдельный namespace `/api/v1/algopack/*`.

Практическая логика такая:

- realtime `candles/orderbook/trades` закрываются обычным ISS-слоем;
- уникальные датасеты ALGOPACK идут через отдельный datashop API.

### `GET /api/v1/algopack/dataset`

Универсальный endpoint для датасетов ALGOPACK вида:

```text
/iss/datashop/algopack/{market}/{dataset}
```

Пример:

```bash
curl "http://127.0.0.1:8080/api/v1/algopack/dataset?market=eq&dataset=obstats&date=2024-10-15"
```

### `GET /api/v1/algopack/tradestats`

Пример:

```bash
curl "http://127.0.0.1:8080/api/v1/algopack/tradestats?market=eq&date=2024-10-15"
```

### `GET /api/v1/algopack/orderstats`

Пример:

```bash
curl "http://127.0.0.1:8080/api/v1/algopack/orderstats?market=eq&date=2024-10-15"
```

### `GET /api/v1/algopack/obstats`

Пример:

```bash
curl "http://127.0.0.1:8080/api/v1/algopack/obstats?market=eq&date=2024-10-15"
```

### `GET /api/v1/algopack/hi2`

Пример:

```bash
curl "http://127.0.0.1:8080/api/v1/algopack/hi2?market=eq&date=2024-10-15"
```

### `GET /api/v1/algopack/futoi`

Пример:

```bash
curl "http://127.0.0.1:8080/api/v1/algopack/futoi"
```

### `GET /api/v1/algopack/realtime/candles`

Alias на realtime свечи.

Пример:

```bash
curl "http://127.0.0.1:8080/api/v1/algopack/realtime/candles?ticker=SBER&timeframe=1H&from=2024-01-01&to=2024-01-10"
```

### `GET /api/v1/algopack/realtime/orderbook`

Alias на realtime стакан.

```bash
curl "http://127.0.0.1:8080/api/v1/algopack/realtime/orderbook?security=SBER&depth=20"
```

### `GET /api/v1/algopack/realtime/trades`

Alias на realtime сделки.

```bash
curl "http://127.0.0.1:8080/api/v1/algopack/realtime/trades?security=SBER&limit=100"
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
- auth-настройки для MOEX ISS / AlgoPack;
- адрес ML сервиса;
- настройки LLM analyzer;
- веса агрегатора;
- риск-лимиты;
- пути к raw storage и decision log.

### Настройки MOEX / AlgoPack

Backend сейчас поддерживает несколько способов авторизации:

- `moex.username` + `moex.password` — basic auth;
- `moex.passport_cert` — cookie `MicexPassportCert`;
- `moex.bearer_token` — bearer token;
- `moex.api_key_header` + `moex.api_key_value` — кастомный header с ключом.

Разделение base URL:

- `moex.base_url` — обычный ISS, по умолчанию `https://iss.moex.com`
- `moex.algopack_base_url` — datashop / ALGOPACK, по умолчанию `https://apim.moex.com`

Есть env overrides:

- `MOEX_BASE_URL`
- `MOEX_ALGOPACK_BASE_URL`
- `MOEX_ENGINE`
- `MOEX_MARKET`
- `MOEX_BOARD`
- `MOEX_USERNAME`
- `MOEX_PASSWORD`
- `MOEX_PASSPORT_CERT`
- `MOEX_BEARER_TOKEN`
- `MOEX_API_KEY_HEADER`
- `MOEX_API_KEY_VALUE`
- `MOEX_USER_AGENT`

Практически:

- если у вас логин/пароль от MOEX Passport, используйте `username/password`;
- если у вас уже есть `MicexPassportCert`, используйте `passport_cert`;
- если у вас APIKEY ALGOPACK, используйте `bearer_token`;
- если доступ к данным выдан через внешний шлюз с header/token auth, используйте `api_key_*`.

Секреты лучше хранить через env, а не коммитить в YAML.

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
- успешная сборка backend после добавления AlgoPack / ISS integration;
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

### Smoke-test AlgoPack / ISS integration

Backend должен быть запущен.

Проверка универсального proxy:

```bash
curl "http://127.0.0.1:8080/api/v1/moex/iss?path=/iss/securities/SBER"
```

Проверка shortcut endpoint-ов:

```bash
curl "http://127.0.0.1:8080/api/v1/moex/security?security=SBER"
curl "http://127.0.0.1:8080/api/v1/moex/orderbook?security=SBER&depth=20"
curl "http://127.0.0.1:8080/api/v1/moex/trades?security=SBER&limit=100"
curl "http://127.0.0.1:8080/api/v1/moex/sitenews?limit=20"
```

Если вы используете подписочные ресурсы AlgoPack, перед этим нужно заполнить auth-поля MOEX в config или env.

### Smoke-test ALGOPACK datashop

Backend должен быть запущен, а `MOEX_BEARER_TOKEN` должен быть задан.

Примеры:

```bash
curl "http://127.0.0.1:8080/api/v1/algopack/obstats?market=eq&date=2024-10-15"
curl "http://127.0.0.1:8080/api/v1/algopack/orderstats?market=eq&date=2024-10-15"
curl "http://127.0.0.1:8080/api/v1/algopack/tradestats?market=eq&date=2024-10-15"
curl "http://127.0.0.1:8080/api/v1/algopack/hi2?market=eq&date=2024-10-15"
curl "http://127.0.0.1:8080/api/v1/algopack/futoi"
```

Если токен рабочий, backend должен вернуть JSON с `request_url`, `path` и `data`.

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

## Что ещё нужно для реального AlgoPack доступа

Чтобы проверить не только публичные ISS ресурсы, но и подписочные AlgoPack данные, в backend нужно подставить один из реальных форматов доступа:

- `MOEX_USERNAME` и `MOEX_PASSWORD`;
- или `MOEX_PASSPORT_CERT`;
- или `MOEX_BEARER_TOKEN`, если у вас APIKEY ALGOPACK;
- или точный header/token формат, если у вас доступ выдан не basic/cookie, а через отдельный key gateway.

Сейчас backend уже умеет эти варианты применять, но реальные секреты в репозиторий не добавлены.

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
