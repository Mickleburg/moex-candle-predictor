# ML-реализация для предсказания свечей MOEX

Документ описывает ML-часть проекта `moex-candle-predictor`.

## Обзор

ML pipeline предсказывает движения цен свечей через multiclass classification:
- **Вход**: Исторические OHLCV свечи
- **Выход**: Класс токена (0-6), представляющий нормализованный возврат на горизонте h=3
- **Модель**: LightGBM multiclass classifier (MVP)
- **Inference**: FastAPI сервис на порту 8001

## Структура папки ml/

```
ml/
├── configs/              # Конфигурационные файлы
│   ├── data.yaml        # Загрузка и разделение данных
│   ├── features.yaml    # Feature engineering и токенизация
│   ├── train.yaml       # Обучение модели
│   └── eval.yaml        # Настройки оценки
├── src/
│   ├── data/           # Загрузка, чистка, split
│   ├── features/       # Признаки, токенизация, окна
│   ├── models/         # Реализации моделей
│   ├── evaluation/     # Метрики и бэктест
│   ├── service/        # Inference API
│   └── utils/          # I/O и конфиги
├── artifacts/          # Сохранённые артефакты
│   ├── model.pkl       # Обученная модель
│   ├── tokenizer.pkl   # Обученный tokenizer
│   └── metadata.json   # Метаданные обучения
└── requirements.txt    # Python зависимости
```

## Установка

```bash
cd ml
pip install -r requirements.txt
```

## Обучение

### Предварительные требования

- Сырые свечи в `data/raw/` (формат Parquet)
- Минимум 1000+ свечей для обучения

### Запуск обучения

```bash
cd ml
python -m src.models.train --config-dir configs
```

### Конфигурация

Отредактируйте `ml/configs/*.yaml`:
- `data.yaml`: источник данных, tickers, timeframes, split ratios
- `features.yaml`: токенизация (K=7, h=3), размер окна (L=32), индикаторы
- `train.yaml`: тип модели, гиперпараметры LightGBM, пути к артефактам

### Артефакты

После обучения артефакты сохраняются в `ml/artifacts/`:
- `model.pkl` — обученная модель LightGBM
- `tokenizer.pkl` — обученный квантильный tokenizer
- `metadata.json` — метаданные обучения (ticker, timeframe, метрики)

## Inference сервис

### Запуск сервиса

```bash
cd ml
uvicorn src.service.api:app --host 0.0.0.0 --port 8001
```

### Проверка health

```bash
curl http://localhost:8001/health
```

Ответ:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "2024-01-01T12:00:00.000000",
  "timestamp": "2024-01-01T12:00:00.000000"
}
```

### Предсказание

```bash
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d @request.json
```

**Запрос (`request.json`):**
```json
{
  "candles": [
    {
      "begin": "2024-01-01T10:00:00",
      "open": 250.0,
      "high": 251.5,
      "low": 249.0,
      "close": 250.5,
      "volume": 150000,
      "ticker": "SBER",
      "timeframe": "1H"
    }
  ]
}
```

**Ответ:**
```json
{
  "predicted_token": 4,
  "probabilities": [0.1, 0.05, 0.1, 0.15, 0.3, 0.2, 0.1],
  "action": "buy",
  "confidence": 0.3,
  "model_version": "2024-01-01T12:00:00.000000",
  "ticker": "SBER",
  "timeframe": "1H",
  "timestamp": "2024-01-01T12:00:00.000000",
  "n_candles_used": 32,
  "diagnostics": {
    "window_size": 32,
    "n_classes": 7,
    "horizon": 3
  }
}
```

**Примечание:** API требует минимум 32 свечи для prediction window.

## Краткое описание pipeline

1. **Load**: Загрузка свечей из `data/raw/` (Parquet)
2. **Clean**: Валидация OHLCV, сортировка по времени, удаление дубликатов
3. **Split**: Time-based train/val/test split (70/15/15)
4. **Features**: Вычисление технических индикаторов (returns, ATR, volatility, EMA, candle features, time features)
5. **Tokenize**: Fit квантильного tokenizer только на train, transform на всех split'ах
6. **Windows**: Построение скользящих окон (L=32) с target на горизонте h=3
7. **Train**: Обучение LightGBM с early stopping
8. **Evaluate**: Вычисление classification и trading метрик

## Доступные модели

- `majority`: Baseline — самый частый класс
- `markov`: Baseline — Markov chain
- `lgbm`: LightGBM multiclass classifier (основной MVP)
- `rnn`: RNN stub (будущая реализация)

## Метрики

**Classification:** Accuracy, macro-F1, weighted-F1, precision, recall, log-loss, confusion matrix

**Trading:** Sharpe ratio, Max drawdown, PnL, win rate

## Защита от leakage

- Tokenizer fit только на train данных
- Rolling индикаторы используют только исторические данные
- Target использует будущие данные на горизонте h=3
- Time split валидирует хронологический порядок
- Inference использует тот же код preprocessing, что и training

## Smoke tests

Базовые проверки работоспособности:

```bash
cd ml
python test_smoke.py
```

## Ограничения MVP

- RNN не реализован (только stub)
- Time split вместо walk-forward validation
- Простые признаки (только технические индикаторы)
- Классификация вместо регрессии (квантили теряют точность magnitude)
- Нет автоматического hyperparameter tuning
- Простое versioning по timestamp

## Подробная документация

Подробная архитектура, API контракт и эксперименты описаны в:
- `docs/architecture.md` — архитектура проекта и data flow
- `docs/api_contract.md` — API контракт для inference сервиса
- `docs/experiments.md` — как запускать эксперименты и интерпретировать результаты

## Зависимости

См. `ml/requirements.txt`. Основные зависимости:
- pandas, numpy — работа с данными
- lightgbm — основная модель
- scikit-learn — метрики и preprocessing
- fastapi, uvicorn — API сервер
- pydantic — валидация запросов/ответов
- pyarrow — Parquet I/O
- pyyaml — конфиги
