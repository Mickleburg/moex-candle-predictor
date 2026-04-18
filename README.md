# MOEX Candle Predictor

Проект предсказывает движения цен свечей на бирже MOEX через подход "свечи как токены".

## Идея решения

Исторические свечи представляются как последовательность токенов, и задача — предсказать следующий токен в последовательности (аналогично языковым моделям).

**MVP параметры:**
- Ticker: SBER
- Timeframe: 1H
- Target: класс движения через h=3 свечи вперёд
- Количество классов: K=7 (квантильные бины)
- Размер окна: L=32 свечей
- Модель: LightGBM (CPU-optimized)
- Inference: FastAPI сервис

## Текущий статус проекта

**Готово:**
- ML pipeline для обучения моделей (data loading, feature engineering, tokenization, training, evaluation)
- Baseline модели: Majority, Markov, LogisticRegression, LightGBM
- Inference сервис (FastAPI) с API контрактами
- Evaluation layer (classification metrics, backtest, online evaluation)
- Документация (architecture, API contract, experiments)

**В разработке:**
- Backend (Go) для интеграции с внешними источниками данных и управления торговлей
- Walk-forward validation вместо time split
- RNN implementation для sequence modeling

## Структура репозитория

```
moex-candle-predictor/
├── backend/              # Go backend (в разработке)
├── data/                 # Хранилище данных
│   ├── raw/             # Сырые свечи (Parquet)
│   ├── processed/       # Очищенные данные
│   ├── predictions/     # Предсказания (online)
│   └── reports/        # Отчёты (backtest, metrics)
├── ml/                  # ML implementation
│   ├── configs/         # Конфигурации YAML
│   ├── src/            # Исходный код ML
│   ├── artifacts/      # Сохранённые артефакты
│   ├── requirements.txt
│   └── README.md       # Документация ML части
├── shared/              # Общие схемы
│   └── schemas/        # JSON schemas
├── docs/               # Документация
│   ├── architecture.md
│   ├── api_contract.md
│   └── experiments.md
└── README.md           # Этот файл
```

## Что уже реализовано

### ML Pipeline
- Загрузка свечей из Parquet
- Валидация и чистка данных
- Time-based split (train/val/test)
- Технические индикаторы (returns, ATR, volatility, EMA, candle features, time features)
- Квантильная токенизация (fit только на train)
- Построение скользящих окон
- Обучение LightGBM с early stopping
- Classification metrics (accuracy, F1, precision, recall, log-loss, confusion matrix)
- Backtest с trading metrics (Sharpe, MDD, PnL, win rate)

### Inference Service
- FastAPI сервис на порту 8001
- Endpoints: GET /health, POST /predict
- Pydantic валидация запросов/ответов
- Загрузка артефактов и preprocessing

### Documentation
- `docs/architecture.md` — архитектура, data flow, leakage prevention
- `docs/api_contract.md` — API контракт, примеры на Go
- `docs/experiments.md` — как запускать эксперименты, интерпретация результатов

## Быстрый старт для ML-части

### 1. Установка зависимостей

```bash
cd ml
pip install -r requirements.txt
```

### 2. Обучение модели

```bash
cd ml
python -m src.models.train --config-dir configs
```

Артефакты сохранятся в `ml/artifacts/`.

### 3. Запуск inference сервиса

```bash
cd ml
uvicorn src.service.api:app --host 0.0.0.0 --port 8001
```

### 4. Проверка

```bash
curl http://localhost:8001/health
```

Подробная инструкция в `ml/README.md`.

## Документация

- **Архитектура проекта:** `docs/architecture.md` — высокоуровневая архитектура, data flow, связи между модулями, утечка данных
- **API контракт:** `docs/api_contract.md` — endpoints, форматы запросов/ответов, примеры на Go, правила совместимости
- **Эксперименты:** `docs/experiments.md` — как запускать обучение, интерпретация метрик, reproducible run
- **ML часть:** `ml/README.md` — практический старт для работы с ML pipeline

## Ограничения MVP

- Time split вместо walk-forward validation
- Простые признаки (только технические индикаторы)
- Классификация вместо регрессии (квантили теряют точность magnitude)
- RNN не реализован (только stub)
- Нет автоматического hyperparameter tuning
- Backend (Go) в разработке

## Что будет добавлено дальше

**Краткосрочные:**
- Walk-forward validation
- Больше признаков (order book, market microstructure)
- Hyperparameter tuning (Optuna)
- Model versioning (MLflow)

**Среднесрочные:**
- RNN implementation (PyTorch/TensorFlow)
- Ensemble models (LightGBM + RNN)
- Online learning
- Monitoring (Prometheus/Grafana)

**Backend:**
- Интеграция с MOEX API
- Управление торговлей
- Логирование и мониторинг

## Маршрут чтения проекта

Для понимания проекта рекомендуется следующий порядок:

1. **README.md** (этот файл) — общий обзор проекта
2. **ml/README.md** — практический старт для ML части
3. **docs/architecture.md** — архитектура, data flow, утечка данных
4. **docs/experiments.md** — как запускать эксперименты
5. **docs/api_contract.md** — API контракт для интеграции с backend

## Важные каталоги для старта

- `ml/configs/` — конфигурации для обучения (data.yaml, features.yaml, train.yaml)
- `ml/artifacts/` — сохранённые модели после обучения
- `data/raw/` — сырые свечи в формате Parquet
- `shared/schemas/` — JSON схемы для контрактов между backend и ML