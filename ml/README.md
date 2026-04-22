# ML

`ml/` содержит Python-часть проекта: training pipeline и FastAPI inference service.

## Назначение ML-части

Текущая подтверждаемая ответственность `ml/`:

- загрузить raw candles из `data/raw/`;
- очистить и отсортировать данные;
- выполнить time split;
- вычислить признаки;
- построить target tokens;
- обучить модель;
- сохранить артефакты;
- поднять inference API, который принимает массив свечей и возвращает prediction.

ML-часть не должна перекладываться в backend и не должна вызываться через прямой импорт из Go-кода.

## Структура Модуля

- `configs/` — YAML-конфиги для data/features/train/eval.
- `src/data/` — загрузка, очистка, splitting.
- `src/features/` — indicators, tokenizer, window builders.
- `src/models/` — baseline models, LightGBM wrapper, training entrypoint.
- `src/service/` — FastAPI app, predictor, Pydantic schemas.
- `src/evaluation/` — helper-функции для metrics/backtest/online evaluation.
- `artifacts/` — checked-in model/tokenizer/metadata.
- `notebooks/` — exploratory notebooks, не часть automated runtime path.

## Ключевые Файлы

- `src/models/train.py` — основной training orchestrator.
- `src/data/load.py` — загрузка raw candles из файла или директории.
- `src/data/clean.py` — очистка, сортировка, дедупликация, отбрасывание invalid candles.
- `src/data/split.py` — time-based train/val/test split.
- `src/features/indicators.py` — feature engineering по OHLCV.
- `src/features/tokenizer.py` — квантильная токенизация будущих returns.
- `src/features/windows.py` — build tabular/inference windows.
- `src/models/lgbm_model.py` — основной checked-in model wrapper.
- `src/service/api.py` — FastAPI entrypoint.
- `src/service/predictor.py` — загрузка артефактов и inference preprocessing.
- `src/service/schemas.py` — request/response Pydantic schemas для API.

## Training Flow

Точка входа:

```powershell
Set-Location C:\Users\ancha\Projects\MOEX\moex-candle-predictor\ml
python -m src.models.train --config-dir configs
```

Что делает pipeline:

1. загружает конфиги через `src/utils/config.py`;
2. читает raw data из `data_config["raw_data_path"]`;
3. фильтрует по первому ticker и первому timeframe из config;
4. вызывает `clean_candles`;
5. делает time split на train/val/test;
6. на каждом split считает indicators;
7. fit-ит tokenizer на train split и transform-ит val/test;
8. строит tabular windows;
9. обучает выбранную модель;
10. считает classification metrics;
11. сохраняет `model.pkl`, `tokenizer.pkl`, `metadata.json`.

Поддерживаемые `model_type` по коду:

- `majority`
- `markov`
- `logistic`
- `lgbm`

`rnn` присутствует в кодовой базе как future/stub path, но не подключен в основной pipeline как рабочий вариант.

## Inference Flow

Точка входа:

```powershell
Set-Location C:\Users\ancha\Projects\MOEX\moex-candle-predictor\ml
uvicorn src.service.api:app --host 127.0.0.1 --port 8001
```

Поддерживаемые endpoints:

- `GET /health`
- `POST /predict`

`src/service/api.py` поднимает FastAPI app и лениво инициализирует `CandlePredictor`.

`CandlePredictor` делает:

1. загрузку `metadata.json`;
2. загрузку `model.pkl`;
3. опциональную загрузку `tokenizer.pkl`;
4. преобразование входных candles в DataFrame;
5. вычисление indicators;
6. построение последнего inference window;
7. `predict` и `predict_proba`;
8. маппинг predicted class в `buy` / `sell` / `hold`.

## Feature Pipeline

По текущему коду training/inference feature engineering строится вокруг:

- returns: `return_1`, `return_3`, `return_5`
- `atr`
- `rolling_volatility`
- `volume_ratio`
- candle body/range/wick features
- EMA distance features
- time features: `hour`, `day_of_week`, `month`

Tokenizer строит target token через нормализованный future return:

- horizon по умолчанию: `3`
- число классов по умолчанию: `7`
- нормализация: ATR-based quantile binning

Window length по умолчанию:

- `window_size = 32`

## Predict Path

`POST /predict` принимает массив candles.

Минимальные runtime assumptions по коду:

- свечей должно быть не меньше `L`, где `L` берется из `metadata["L"]` или fallback `32`;
- свечи должны быть совместимы по колонкам с `_candles_to_dataframe`;
- request подразумевает один ticker и один timeframe;
- inference использует последние `window_size` свечей как контекст.

Shared schemas описывают только форму request/response, но не покрывают всю preprocessing semantics.

## Configs And Artifacts

Используемые config files:

- `configs/data.yaml`
- `configs/features.yaml`
- `configs/train.yaml`
- `configs/eval.yaml`

Фактически используемые artifact files:

- `artifacts/model.pkl`
- `artifacts/tokenizer.pkl`
- `artifacts/metadata.json`

По `metadata.json` сейчас видно:

- ticker: `SBER`
- timeframe: `1H`
- horizon: `3`
- `K = 7`
- `L = 32`
- artifact version: `2026-04-19T17:29:17.793124`

## Ограничения И Caveats

- `data/raw/` в репозитории сейчас не содержит reproducible dataset, поэтому training path документируется по коду, а не по проверенному повторному запуску.
- Checked-in `.venv` не должен считаться переносимым окружением.
- Часть evaluation/backtest helper-кода существует, но основной training entrypoint не пишет полноценный backtest report автоматически.
- Не все параметры из `features.yaml` и `eval.yaml` напрямую влияют на фактический кодовый путь.
- Shared schemas есть только для inference API, а не для training raw contract.

## Статус Checked-In Artifacts

Checked-in artifacts могут быть устаревшими относительно текущего кода.

Это важно по нескольким причинам:

- pipeline уже менялся после leakage-related исправлений;
- `metadata.json` показывает feature set, который нельзя автоматически считать синхронным с текущим feature engineering без retrain;
- в `data/reports/decision_log.jsonl` есть исторические записи о feature mismatch (`800` vs `832` features), что прямо указывает на риск train/serve drift.

Поэтому:

- наличие `model.pkl` и `metadata.json` не доказывает консистентность;
- успешный import/load артефактов не равен корректности предсказаний;
- для надежной синхронизации нужен отдельный retrain и повторная verification.
