# ML

`ml/` содержит pipeline подготовки данных, обучение модели и inference API.

## Структура

- `configs/` — YAML-конфиги данных, признаков, обучения и evaluation.
- `src/data/` — загрузка, чистка, split.
- `src/features/` — индикаторы, tokenizer, окна.
- `src/models/` — baseline-модели, LightGBM wrapper, training pipeline.
- `src/service/` — FastAPI inference.
- `artifacts/` — модель, tokenizer, metadata.

## Training pipeline

Точка входа:

```bash
cd ml
python -m src.models.train --config-dir configs
```

Что делает pipeline:

1. читает raw candles из `data/raw`;
2. чистит и сортирует данные;
3. делает time split на train/val/test;
4. считает признаки;
5. fit/transform tokenizer;
6. строит окна;
7. обучает выбранную модель;
8. сохраняет артефакты в `ml/artifacts/`.

Поддерживаемые `model_type` в коде:

- `majority`
- `markov`
- `logistic`
- `lgbm`

`rnn` присутствует как stub и не готов к использованию.

## Inference API

Запуск:

```bash
cd ml
uvicorn src.service.api:app --host 127.0.0.1 --port 8001
```

Endpoints:

- `GET /health`
- `POST /predict`

`/predict` принимает массив свечей и возвращает `predicted_token`, `probabilities`, `action`, `confidence`, `model_version` и diagnostics.

## Ограничения

- Конфиги `features.yaml` и `eval.yaml` описывают желаемое поведение шире, чем реально используется кодом; не все параметры участвуют в pipeline.
- `artifacts/metadata.json` описывает артефакты, но не заменяет проверку совместимости train/serve.
- В репозитории есть серьёзные риски leakage и несогласованности train/inference логики; перед production это требует отдельного исправления и переобучения.
