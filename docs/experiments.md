# Эксперименты и обучение ML моделей

## Как запускать первый ML эксперимент

### Предварительные требования

1. **Данные:** Сырые свечи должны быть в `data/raw/` в формате Parquet.
   - Минимум 1000+ свечей для обучения
   - Формат: Parquet с колонками (ticker, timeframe, begin, end, open, high, low, close, volume, value, source)

2. **Установка зависимостей:**
```bash
cd ml
pip install -r requirements.txt
```

3. **Конфигурация:** Проверьте `ml/configs/*.yaml`:
   - `data.yaml`: путь к raw data
   - `features.yaml`: K, h, L, параметры индикаторов
   - `train.yaml`: model type, гиперпараметры
   - `eval.yaml`: evaluation settings

### Запуск обучения

```bash
cd ml
python -m src.models.train --config-dir configs
```

**Что происходит:**
1. Загружаются конфигурации из `configs/`
2. Загружаются сырые свечи из `data/raw/`
3. Данные чистятся и разделяются по времени (train/val/test)
4. Считаются технические индикаторы
5. Tokenizer fit только на train, transform на val/test
6. Строятся скользящие окна (L=32, h=3)
7. Обучается модель (LightGBM по умолчанию)
8. Считаются метрики на val/test
9. Сохраняются артефакты в `ml/artifacts/`

### Проверка результатов

После обучения проверьте:
- `ml/artifacts/model.pkl` — обученная модель
- `ml/artifacts/tokenizer.pkl` — обученный tokenizer
- `ml/artifacts/metadata.json` — метаданные и метрики
- `data/reports/` — отчёты по бэктесту (если включён)

## Конфигурационные файлы

### ml/configs/data.yaml

**За что отвечает:** Загрузка и разделение данных.

```yaml
# Путь к сырым данным (относительно ml/ каталога)
raw_data_path: "../../data/raw"

# Формат данных (parquet или csv)
data_format: "parquet"

# Tickers и timeframes (extensible)
tickers:
  - "SBER"
timeframes:
  - "1H"

# Обязательные колонки в данных
required_columns:
  - "ticker"
  - "timeframe"
  - "begin"
  - "end"
  - "open"
  - "high"
  - "low"
  - "close"
  - "volume"
  - "value"
  - "source"

# Time split ratios
train_ratio: 0.7      # 70% для обучения
val_ratio: 0.15       # 15% для валидации
test_ratio: 0.15      # 15% для теста
min_train_size: 1000  # Минимум 1000 свечей в train
```

### ml/configs/features.yaml

**За что отвечает:** Feature engineering и токенизация.

```yaml
# Токенизация
num_classes: 7   # K=7 классов (квантильные бины)
horizon: 3       # h=3 свечи вперёд для target

# Окна
window_size: 32  # L=32 свечей в окне

# Технические индикаторы
indicators:
  returns:
    periods: [1, 3, 5]  # 1, 3, 5 периодные returns
  
  atr:
    period: 14           # ATR с периодом 14
  
  rolling_volatility:
    window: 20           # Rolling volatility с окном 20
  
  volume_ratio:
    window: 20           # Volume ratio к mean за 20 периодов
  
  ema:
    periods: [9, 20, 50] # EMA с периодами 9, 20, 50
  
  candle_features:
    - "body"             # Тело свечи
    - "range"             # Размах свечи
    - "upper_wick"        # Верхняя тень
    - "lower_wick"        # Нижняя тень
    - "body_ratio"        # Отношение тела к размаху

# Time features
time_features:
  - "hour"              # Час дня
  - "day_of_week"       # День недели
  - "month"             # Месяц

# Feature normalization (fit на train only)
# Примечание: tokenizer использует ATR для нормализации, эта настройка резервируется для будущего
normalization:
  method: "standard"     # standard, minmax, или none (резервируется для будущего)
```

### ml/configs/train.yaml

**За что отвечает:** Обучение модели и сохранение артефактов.

```yaml
# Выбор модели
model_type: "lgbm"  # majority, markov, lgbm

# Random seed для воспроизводимости
random_state: 42

# Гиперпараметры LightGBM
lgbm_params:
  objective: "multiclass"
  num_class: 7
  n_estimators: 100
  max_depth: 6
  learning_rate: 0.1
  num_leaves: 31
  min_child_samples: 20
  subsample: 0.8
  colsample_bytree: 0.8
  reg_alpha: 0.1
  reg_lambda: 0.1
  verbose: -1

# Early stopping
early_stopping_rounds: 10
verbose: 1

# Пути (относительно ml/ каталога)
artifacts_dir: "artifacts"
reports_dir: "../../data/reports"
predictions_dir: "../../data/predictions"
```

### ml/configs/eval.yaml

**За что отвечает:** Настройки оценки и бэктеста.

```yaml
# Metrics to compute
metrics:
  classification:
    - "accuracy"
    - "macro_f1"
    - "weighted_f1"
    - "log_loss"
    - "confusion_matrix"
  
  trading:
    - "sharpe_ratio"
    - "max_drawdown"
    - "total_pnl"
    - "win_rate"

# Backtesting settings
backtest:
  initial_capital: 100000.0
  commission: 0.0005  # 0.05% per trade
  position_size: 1.0

# Paths
reports_dir: "../../data/reports"
predictions_dir: "../../data/predictions"
```

## Доступные baseline модели

### 1. MajorityClassifier

**Описание:** Предсказывает самый частый класс в train set.

**Использование:**
```yaml
# В train.yaml
model_type: "majority"
```

**Когда использовать:** Для проверки, что модель вообще может лучше случайного угадывания.

### 2. MarkovClassifier

**Описание:** Предсказывает следующий токен на основе переходных вероятностей (n-gram, n=1).

**Использование:**
```yaml
# В train.yaml
model_type: "markov"
```

**Когда использовать:** Для проверки, что простая последовательная модель может быть полезной.

### 3. LogisticRegressionBaseline

**Описание:** Logistic regression на табличных признаках.

**Использование:**
```yaml
# В train.yaml
model_type: "logistic"
```

**Когда использовать:** Для сравнения с LightGBM (простая линейная модель vs нелинейная).

### 4. LGBMClassifier (MVP)

**Описание:** LightGBM multiclass classifier с balanced class weights и early stopping.

**Использование:**
```yaml
# В train.yaml
model_type: "lgbm"
```

**Когда использовать:** Основной MVP модель — лучшее качество на табличных данных, CPU-optimized.

### 5. RNNClassifier (Stub)

**Описание:** RNN stub для будущей реализации (требует PyTorch/TensorFlow).

**Статус:** Не реализован в MVP — только интерфейс с NotImplementedError.

## Как устроена токенизация

### Идея

Токенизация квантильными бинами на нормализованных возвратах.

### Формула

1. **Нормализованный возврат:**
   ```
   normalized_return = return_h / ATR
   ```
   - `return_h`: возврат через h=3 свечи вперёд
   - `ATR`: Average True Range (волатильность)

2. **Квантильная бинизация:**
   ```
   token = quantile_bin(normalized_return, K=7)
   ```
   - Fit только на train данных
   - K=7 бинов (0-6)
   - Bin edges сохраняются для inference

### Код

```python
# В tokenizer.py
class CandleTokenizer:
    def fit(self, df):
        # Вычисляем normalized returns
        normalized_returns = self._compute_normalized_returns(df)
        
        # Fit K=7 quantile bins
        self.bin_edges = np.quantile(
            normalized_returns,
            np.linspace(0, 1, self.n_bins + 1)
        )
        
    def transform(self, df):
        # Вычисляем normalized returns
        normalized_returns = self._compute_normalized_returns(df)
        
        # Применяем bin edges
        tokens = np.digitize(normalized_returns, self.bin_edges) - 1
        
        # Clip to [0, K-1]
        tokens = np.clip(tokens, 0, self.n_bins - 1)
```

### Почему так

- **Robust к выбросам:** Квантили адаптивны к распределению
- **Нормализация:** ATR нормализует по волатильности
- **Интерпретируемость:** Токены 0-6 соответствуют quantile returns (0=сильно вниз, 6=сильно вверх)

## Какие признаки используются в MVP

### Технические индикаторы

| Группа | Признаки | Описание |
|--------|----------|----------|
| Returns | return_1, return_3, return_5 | Close-to-close returns за 1, 3, 5 периодов |
| ATR | atr | Average True Range (14 период) |
| Volatility | rolling_volatility | Rolling std returns (20 период) |
| Volume | volume_ratio | Current / mean volume (20 период) |
| EMA | ema_9, ema_20, ema_50 | EMA с периодами 9, 20, 50 |
| Candle | body, range, upper_wick, lower_wick, body_ratio | Форма свечи |
| Time | hour, day_of_week, month | Временные признаки |

### Всего

Около 20+ признаков на свече после feature engineering.

### Код

```python
# В indicators.py
def compute_all_indicators(df):
    # Returns
    df["return_1"] = df["close"].pct_change(1)
    df["return_3"] = df["close"].pct_change(3)
    df["return_5"] = df["close"].pct_change(5)
    
    # ATR
    df["atr"] = compute_atr(df, period=14)
    
    # Rolling volatility
    df["rolling_volatility"] = df["return_1"].rolling(20).std()
    
    # Volume ratio
    df["volume_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
    
    # EMA
    df["ema_9"] = df["close"].ewm(span=9).mean()
    df["ema_20"] = df["close"].ewm(span=20).mean()
    df["ema_50"] = df["close"].ewm(span=50).mean()
    
    # Candle features
    df["body"] = df["close"] - df["open"]
    df["range"] = df["high"] - df["low"]
    df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
    df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]
    df["body_ratio"] = df["body"] / df["range"]
    
    # Time features
    df["hour"] = df["begin"].dt.hour
    df["day_of_week"] = df["begin"].dt.dayofweek
    df["month"] = df["begin"].dt.month
    
    return df
```

## Как делается time split / walk-forward

### Time Split (MVP)

Используется простой time-based split с фиксированными пропорциями:

```python
# В split.py
def time_split(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    
    # Проверка утечки
    assert train_df["begin"].max() < val_df["begin"].min()
    assert val_df["begin"].max() < test_df["begin"].min()
    
    return train_df, val_df, test_df
```

### Walk-Forward (Future для MVP)

Код WalkForwardSplit реализован в `ml/src/data/split.py`, но не используется в MVP training pipeline. Для production рекомендуется walk-forward validation:
- Sliding window по времени
- Train на [t, t+N], val/test на [t+N, t+N+M]
- Сдвиг окна и повтор

### Почему Time Split для MVP

- **Простота:** Легко реализовать и понимать
- **Reproducible:** Детерминированный split
- **Быстро:** Нет пересечения windows

## Какие метрики считать

### Classification Metrics

| Метрика | Описание | Использование |
|---------|----------|---------------|
| Accuracy | Доля правильных предсказаний | Общая оценка |
| Macro-F1 | F1 score усреднённый по классам | Баланс между precision/recall |
| Weighted-F1 | F1 score взвешенный по support | Учитывает дисбаланс классов |
| Macro Precision | Precision усреднённый по классам | Качество положительных предсказаний |
| Macro Recall | Recall усреднённый по классам | Полнота предсказаний |
| Log Loss | Cross-entropy loss | Вероятностная оценка |
| Confusion Matrix | Матрица ошибок | Анализ ошибок по классам |

### Trading Metrics

| Метрика | Описание | Использование |
|---------|----------|---------------|
| Sharpe Ratio | (mean / std) returns | Risk-adjusted return |
| Max Drawdown | Максимальная просадка | Risk assessment |
| Total PnL | Суммарная прибыль | Абсолютный результат |
| Win Rate | Доля прибыльных сделок | Частота успеха |
| Hit Rate | Доля прибыльных сделок с position | Альтернатива win rate |

### Код

```python
# В metrics.py
def compute_classification_metrics(y_true, y_pred, y_proba):
    metrics = {}
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["macro_f1"] = f1_score(y_true, y_pred, average="macro")
    metrics["weighted_f1"] = f1_score(y_true, y_pred, average="weighted")
    metrics["macro_precision"] = precision_score(y_true, y_pred, average="macro")
    metrics["macro_recall"] = recall_score(y_true, y_pred, average="macro")
    metrics["log_loss"] = log_loss(y_true, y_proba)
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
    return metrics

# В backtest.py
def backtest_strategy(df, predictions, initial_capital=100000, commission=0.0005):
    # Конвертируем predictions в signals (buy/sell/hold)
    signals = predictions_to_signals(predictions)
    
    # Считаем returns
    returns = df["close"].pct_change()
    
    # Strategy returns с commission
    strategy_returns = (signals * returns) - trade_costs
    
    # Metrics
    sharpe = strategy_returns.mean() / strategy_returns.std()
    max_dd = max_drawdown(strategy_returns)
    total_pnl = strategy_returns.sum()
    win_rate = (strategy_returns > 0).mean()
    
    return {"sharpe_ratio": sharpe, "max_drawdown": max_dd, "total_pnl": total_pnl, "win_rate": win_rate}
```

## Как интерпретировать результаты

### Classification Metrics

- **Accuracy > 1/K (1/7 ≈ 14%):** Модель лучше случайного угадывания
- **Macro-F1 > 0.3:** Приемлемое качество для 7 классов
- **Log Loss < 1.5:** Хорошие вероятности
- **Confusion Matrix:** Смотрите, какие классы путаются

### Trading Metrics

- **Sharpe Ratio > 1:** Хороший risk-adjusted return
- **Max Drawdown < 20%:** Приемлемый риск
- **Win Rate > 0.5:** Больше прибыльных сделок чем убыточных

### Пример интерпретации

```
Validation:
  accuracy: 0.35        # Хорошо, лучше случайного (14%)
  macro_f1: 0.32        # Приемлемо для 7 классов
  log_loss: 1.2         # Хорошие вероятности

Test:
  accuracy: 0.32        # Небольшое падение, нормальный overfit
  macro_f1: 0.29        # Приемлемо
  log_loss: 1.3         # Небольшое ухудшение

Backtest:
  sharpe_ratio: 1.5    # Хороший risk-adjusted return
  max_drawdown: 15%     # Приемлемый риск
  total_pnl: 5000       # Положительный результат
  win_rate: 0.55        # Больше прибыльных сделок
```

## Куда пишутся артефакты

### ml/artifacts/

**Файлы:**
- `model.pkl` — Обученная модель (LightGBM)
- `tokenizer.pkl` — Обученный tokenizer (bin edges)
- `metadata.json` — Метаданные обучения

**Metadata включает:**
```json
{
  "ticker": "SBER",
  "timeframe": "1H",
  "horizon": 3,
  "K": 7,
  "L": 32,
  "feature_set": ["return_1", "atr", "ema_9", ...],
  "tokenizer": {"n_bins": 7, "bin_edges": [...], ...},
  "model_class": "LGBMClassifier",
  "train_period": {"start": "...", "end": "...", "n_samples": 1000},
  "val_period": {"start": "...", "end": "...", "n_samples": 200},
  "test_period": {"start": "...", "end": "...", "n_samples": 200},
  "validation_metrics": {"accuracy": 0.35, "macro_f1": 0.32, ...},
  "test_metrics": {"accuracy": 0.32, "macro_f1": 0.29, ...},
  "artifact_version": "2024-01-01T12:00:00.000000"
}
```

### data/predictions/

**Файлы:**
- `online_predictions.parquet` — Live предсказания для online evaluation

**Формат:**
```python
# В online_eval.py
{
  "timestamp": "2024-01-01T12:00:00",
  "prediction": 4,
  "model_version": "...",
  "prob_class_0": 0.1,
  "prob_class_1": 0.05,
  ...
}
```

### data/reports/

**Файлы:**
- `backtest_report.json` — Отчёт по бэктесту

**Формат:**
```json
{
  "total_pnl": 5000,
  "total_return": 0.05,
  "sharpe_ratio": 1.5,
  "max_drawdown": 0.15,
  "n_trades": 100,
  "win_rate": 0.55,
  "hit_rate": 0.52,
  "final_equity": 105000
}
```

## Пример типичного экспериментального сценария

### Шаг 1: Загрузили raw candles

Backend записал свечи в `data/raw/SBER_1H_2024-01.parquet`.

```bash
# Проверяем данные
python -c "import pandas as pd; df = pd.read_parquet('data/raw/SBER_1H_2024-01.parquet'); print(df.head())"
```

### Шаг 2: Обучили baseline модели

**Majority:**
```bash
# Изменяем train.yaml
model_type: "majority"

# Запускаем
python -m src.models.train --config-dir configs
```

**Результат:**
- Val accuracy: ~14% (1/7 случайный)
- Val macro-F1: ~0.1

**Markov:**
```bash
# Изменяем train.yaml
model_type: "markov"

# Запускаем
python -m src.models.train --config-dir configs
```

**Результат:**
- Val accuracy: ~20%
- Val macro-F1: ~0.15

### Шаг 3: Обучили LightGBM

```bash
# Изменяем train.yaml
model_type: "lgbm"

# Запускаем
python -m src.models.train --config-dir configs
```

**Результат:**
- Val accuracy: 35%
- Val macro-F1: 0.32
- Val log-loss: 1.2

### Шаг 4: Сравнили macro-F1 и log-loss

| Model | Val Accuracy | Val Macro-F1 | Val Log-Loss |
|-------|--------------|--------------|---------------|
| Majority | 0.14 | 0.10 | 1.8 |
| Markov | 0.20 | 0.15 | 1.6 |
| LightGBM | 0.35 | 0.32 | 1.2 |

**Вывод:** LightGBM значительно лучше baseline'ов.

### Шаг 5: Посмотрели Sharpe / MDD / PnL

```bash
# В train.py после обучения добавить:
from evaluation import backtest_strategy, save_backtest_report

# Бэктест на test
test_results = backtest_strategy(test_df, test_pred, test_proba)
save_backtest_report(test_results, "data/reports/backtest_report.json")
```

**Результат:**
- Sharpe Ratio: 1.5
- Max Drawdown: 15%
- Total PnL: 5000
- Win Rate: 55%

**Вывод:** Модель даёт положительный risk-adjusted return.

### Шаг 6: Сделали выводы

1. LightGBM значительно лучше baseline'ов (35% vs 14-20% accuracy)
2. Модель даёт положительный PnL с приемлемым риском
3. Ready для деплоя в production

## Минимальный reproducible run

### Шаг 1: Установка

```bash
cd ml
pip install -r requirements.txt
```

### Шаг 2: Генерация mock данных (если нет real data)

```bash
python -c "
from src.data.fixtures import generate_mock_candles
import pandas as pd

df = generate_mock_candles(n=2000, ticker='SBER', timeframe='1H', seed=42)
df.to_parquet('../data/raw/mock_SBER_1H.parquet')
print('Generated mock data')
"
```

### Шаг 3: Обучение

```bash
cd ml
python -m src.models.train --config-dir configs
```

### Шаг 4: Проверка артефактов

```bash
ls ml/artifacts/
# model.pkl
# tokenizer.pkl
# metadata.json
```

### Шаг 5: Запуск inference сервиса

```bash
cd ml
uvicorn src.service.api:app --host 0.0.0.0 --port 8001
```

### Шаг 6: Проверка health

```bash
curl http://localhost:8001/health
```

### Шаг 7: Тестовое предсказание

```bash
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "candles": [
      {"begin": "2024-01-01T10:00:00", "open": 250, "high": 251, "low": 249, "close": 250.5, "volume": 150000}
    ]
  }'
```

## Known Limitations MVP

### 1. Time Split вместо Walk-Forward

**Ограничение:** Фиксированный split не учитывает non-stationarity рынка.

**Влияние:** Модель может переобучиться на конкретный период.

**Future:** Walk-forward validation для production (код WalkForwardSplit существует в split.py, но не используется в MVP).

### 2. Простые признаки

**Ограничение:** Только технические индикаторы, без order book, news, sentiment.

**Влияние:** Ограниченная информативность признаков.

**Future:** Добавление альтернативных источников данных.

### 3. Классификация вместо регрессии

**Ограничение:** Квантили теряют точность magnitude предсказания.

**Влияние:** Меньше информации о силе движения.

**Future:** Hybrid подход (classification + regression).

### 4. Без учёта транзакционных издержек в target

**Ограничение:** Target не учитывает slippage и commission.

**Влияние:** Бэктест может быть оптимистичен.

**Future:** Учёт транзакционных издержек в target.

### 5. RNN не реализован

**Ограничение:** Нет sequence modeling, только tabular.

**Влияние:** Потенциально хуже качество на последовательных паттернах.

**Future:** Реализация RNN/Transformer.

## Next Steps After MVP

### Краткосрочные

1. **Walk-forward validation:** Заменить time split на walk-forward
2. **Feature engineering:** Добавить больше признаков (order book, market microstructure)
3. **Hyperparameter tuning:** Optuna или подобное для автоматического тюнинга
4. **Model versioning:** MLflow для tracking экспериментов

### Среднесрочные

5. **RNN implementation:** PyTorch/TensorFlow для sequence modeling
6. **Ensemble models:** Комбинация LightGBM + RNN
7. **Online learning:** Обновление модели на новых данных
8. **Monitoring:** Prometheus/Grafana для production monitoring

### Долгосрочные

9. **Multi-ticker:** Обучение на нескольких tickers с transfer learning
10. **Multi-timeframe:** Fusion разных таймфреймов
11. **Reinforcement learning:** RL для оптимального execution
12. **Alternative data:** News, sentiment, macro indicators
