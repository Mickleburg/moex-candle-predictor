# Repository Map

## Top-Level Tree

```text
moex-candle-predictor/
├── README.md
├── backend/
├── data/
├── docs/
├── ml/
└── shared/
```

## Top-Level Directories

- `backend/` — Go HTTP backend.
- `data/` — runtime data directories for raw/processed/predictions/reports.
- `docs/` — project documentation for current implemented scope.
- `ml/` — Python training + inference module.
- `shared/` — shared contracts, currently only JSON Schema for `/predict`.

## Root Files

- `README.md` — entry document for current project scope.
- `.env.example` — placeholder env file, currently empty.
- `.gitignore` — repo ignore rules.

## Backend Map

### Dependency Manifests

- `backend/go.mod` — Go module definition for the backend.
- `backend/go.sum` — locked Go dependency checksums.

### Entrypoints

- `backend/cmd/app/main.go` — backend process entrypoint.

### Config

- `backend/config/config.yaml` — default runtime config.
- `backend/internal/config/config.go` — config structs, defaults, env overrides, validation, relative path resolution.

### App Wiring And Routing

- `backend/internal/app/app.go` — app construction, HTTP routes, handlers, middleware.
- `backend/internal/app/db.go` — helper that builds the file-backed store used by the current runtime.

### Models

- `backend/internal/models/candle.go` — candle structs, JSON parsing, defaults, timeframe parsing.
- `backend/internal/models/prediction.go` — ML, decision, risk and health-related DTOs.
- `backend/internal/models/time.go` — flexible time parsing helpers.

### MOEX Clients

- `backend/internal/moex/client.go` — HTTP client to ISS / ALGOPACK.
- `backend/internal/moex/candles.go` — MOEX candle parsing helpers.

### Services

- `backend/internal/service/history_service.go` — validation and persistence orchestration for raw candles.
- `backend/internal/service/predict_service.go` — backend client to ML `/health` and `/predict`.
- `backend/internal/service/decision_service.go` — experimental hybrid-oriented decision orchestration.
- `backend/internal/service/llm_service.go` — heuristic or external LLM analyzer path, hybrid-oriented.
- `backend/internal/service/risk_service.go` — risk gating for decision path.

### Storage

- `backend/internal/storage/files.go` — Parquet raw candle writer and JSONL decision logger.
- `backend/internal/storage/sqlite.go` — SQLite-related placeholder/helper; not the primary current persistence path.

### Docs

- `backend/README.md` — backend-specific documentation.

## ML Map

### Entrypoints

- `ml/src/models/train.py` — training CLI entrypoint.
- `ml/src/service/api.py` — FastAPI inference entrypoint.
- `ml/test_smoke.py` — lightweight smoke checks for imports/config/mock pipeline.

### Configs

- `ml/configs/data.yaml` — data path, required columns, split config.
- `ml/configs/features.yaml` — feature engineering and tokenizer parameters.
- `ml/configs/train.yaml` — model selection and artifact paths.
- `ml/configs/eval.yaml` — metrics/backtest-related config values.

### Data Pipeline

- `ml/src/data/load.py` — load candles from file/directory, normalize columns.
- `ml/src/data/clean.py` — validate/clean candles.
- `ml/src/data/split.py` — time split and walk-forward helpers.
- `ml/src/data/fixtures.py` — mock candle generation for tests/examples.

### Feature Pipeline

- `ml/src/features/indicators.py` — returns, ATR, volatility, volume, EMA, candle/time features.
- `ml/src/features/tokenizer.py` — ATR-normalized quantile tokenizer.
- `ml/src/features/windows.py` — tabular/sequence/inference window builders.
- `ml/src/features/__init__.py` — exported feature API.

### Models

- `ml/src/models/lgbm_model.py` — main LightGBM wrapper.
- `ml/src/models/baseline.py` — majority, markov, logistic baselines.
- `ml/src/models/rnn_model.py` — future/stub model path.
- `ml/src/models/__init__.py` — exported model API and lazy training import.

### Inference Service

- `ml/src/service/api.py` — FastAPI routes and exception handling.
- `ml/src/service/predictor.py` — artifact loading and inference preprocessing.
- `ml/src/service/schemas.py` — Pydantic schemas for `/health` and `/predict`.
- `ml/src/service/__init__.py` — service exports.

### Evaluation Helpers

- `ml/src/evaluation/metrics.py` — classification/trading metrics helpers.
- `ml/src/evaluation/backtest.py` — backtest helper functions.
- `ml/src/evaluation/online_eval.py` — online prediction persistence/evaluation helpers.

### Utils

- `ml/src/utils/config.py` — YAML config loading.
- `ml/src/utils/io.py` — read/write helpers for parquet/csv/json/pickle/joblib.

### Artifacts And Environment

- `ml/artifacts/model.pkl` — checked-in model artifact.
- `ml/artifacts/tokenizer.pkl` — checked-in tokenizer artifact.
- `ml/artifacts/metadata.json` — checked-in metadata and stored metrics.
- `ml/requirements.txt` — Python dependency list.
- `ml/.venv/` — checked-in virtualenv; present in repo but not a reliable portable environment.

### Exploratory Assets

- `ml/notebooks/01_eda.ipynb`
- `ml/notebooks/02_tokenization.ipynb`
- `ml/notebooks/03_baseline.ipynb`

Эти notebooks полезны для exploration, но не входят в основной runtime path.

## Shared Schemas

- `shared/schemas/candles_request.json` — JSON Schema для request `POST /predict`.
- `shared/schemas/prediction_response.json` — JSON Schema для response `POST /predict`.

## Data Directories

- `data/raw/` — raw candle parquet files; сейчас фактически пусто, кроме `.gitkeep`.
- `data/processed/` — placeholder под промежуточные данные.
- `data/predictions/` — placeholder под prediction outputs.
- `data/reports/` — reports and logs.
- `data/reports/decision_log.jsonl` — исторические decision records.

## Docs Directory

- `docs/architecture.md` — current implemented architecture + future boundary.
- `docs/api_contract.md` — backend<->ML contract and raw data contract.
- `docs/runtime_flow.md` — step-by-step runtime flows.
- `docs/ml_module_architecture.md` — detailed ML module structure, candle-language architecture and prediction process.
- `docs/known_limits.md` — verification status and caveats.
- `docs/experiments.md` — training/evaluation scope.
- `docs/repository_map.md` — этот документ.
