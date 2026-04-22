# Runtime Flow

## Flow 1: Ingest / Store Raw Candles

### Trigger Paths

- `POST /api/v1/candles/store`
- `GET /api/v1/candles/fetch`

### Step-By-Step

1. Backend handler принимает JSON batch свечей или сначала загружает candles из MOEX.
2. `HistoryService.PrepareCandles`:
   - проставляет defaults;
   - проверяет OHLCV;
   - сортирует candles;
   - запрещает mixed ticker/timeframe;
   - запрещает duplicate begin и overlap.
3. `FileStore.SaveRawCandles` преобразует batch в parquet rows.
4. Файл записывается в `data/raw/` с именем вида:
   - `SBER_1H_20240101T1000_20240103T0100.parquet`

### Files / Artifacts Touched

- `backend/internal/app/app.go`
- `backend/internal/service/history_service.go`
- `backend/internal/storage/files.go`
- `data/raw/*.parquet`

## Flow 2: Train Pipeline

### Entrypoint

- `ml/src/models/train.py`

### Step-By-Step

1. Загружаются `ml/configs/*.yaml`.
2. `load_and_prepare_data` читает `data/raw/`.
3. `load_candles` объединяет parquet/csv inputs и нормализует названия колонок.
4. `clean_candles`:
   - нормализует timestamps;
   - сортирует;
   - удаляет дубликаты;
   - удаляет invalid candles;
   - обрабатывает missing values.
5. `time_split` делит данные на train/val/test.
6. `compute_all_indicators` строит feature dataframe.
7. `CandleTokenizer` fit-ится на train split и transform-ит val/test.
8. `build_tabular_windows` строит `X` и `y`.
9. `train_model` обучает выбранный model type.
10. `evaluate_model` считает metrics для val/test.
11. `build_metadata` собирает metadata.
12. `save_artifacts` пишет artifacts.

### Files / Artifacts Touched

- `ml/configs/data.yaml`
- `ml/configs/features.yaml`
- `ml/configs/train.yaml`
- `ml/configs/eval.yaml`
- `ml/src/data/load.py`
- `ml/src/data/clean.py`
- `ml/src/data/split.py`
- `ml/src/features/indicators.py`
- `ml/src/features/tokenizer.py`
- `ml/src/features/windows.py`
- `ml/src/models/train.py`
- `ml/artifacts/model.pkl`
- `ml/artifacts/tokenizer.pkl`
- `ml/artifacts/metadata.json`

## Flow 3: Inference Service

### Entrypoint

- `ml/src/service/api.py`

### Step-By-Step

1. FastAPI app стартует.
2. На startup вызывается `get_predictor()`.
3. `CandlePredictor.load()` читает:
   - `ml/artifacts/metadata.json`
   - `ml/artifacts/model.pkl`
   - `ml/artifacts/tokenizer.pkl` при наличии
4. `GET /health` возвращает статус загрузки модели.
5. `POST /predict`:
   - получает candles;
   - преобразует их в DataFrame;
   - проверяет минимум по `L`;
   - считает indicators;
   - собирает последнее окно длины `window_size`;
   - запускает model inference;
   - возвращает prediction payload.

### Files / Artifacts Touched

- `ml/src/service/api.py`
- `ml/src/service/predictor.py`
- `ml/src/service/schemas.py`
- `ml/artifacts/model.pkl`
- `ml/artifacts/tokenizer.pkl`
- `ml/artifacts/metadata.json`

## Flow 4: Backend -> ML Request Path

### Trigger Path

- backend health aggregation;
- decision-oriented backend path, когда нужен ML signal.

### Step-By-Step

1. Backend создает `PredictService`.
2. При необходимости вызывает `GET /health`.
3. Если `strict_health_check` включен и ML unhealthy, backend не считает ML signal usable.
4. Backend формирует `PredictRequest` из candles.
5. `PredictService.doPredict` отправляет `POST /predict`.
6. Backend читает `PredictResponse`.
7. При `4xx` retry не делается.
8. При сетевой ошибке или `5xx` возможен ограниченный retry.
9. При окончательной неудаче decision path уходит в fallback `HOLD`.

### Files / Artifacts Touched

- `backend/internal/service/predict_service.go`
- `backend/internal/models/candle.go`
- `backend/internal/models/prediction.go`
- `backend/config/config.yaml`
- `shared/schemas/candles_request.json`
- `shared/schemas/prediction_response.json`

## Verification Boundary

Эти flows выведены из кода и checked-in artifacts. Они не являются подтверждением того, что каждый flow был заново успешно воспроизведен в этой среде во время данной сверки.
