# API Contract для ML Inference Service

## Зачем нужен Inference API

ML inference API предоставляет HTTP-интерфейс для получения предсказаний модели в реальном времени. Backend (Go) вызывает этот API для получения торговых сигналов на основе последних свечей.

**Основные цели:**
- Разделение ML-логики и backend-логики
- Лёгкое деплоймент и масштабирование ML сервиса
- Валидация запросов/ответов через Pydantic
- Независимость от языка реализации ML (Python)

## Endpoints

### GET /health

**Назначение:** Проверка состояния сервиса и загруженности модели.

**Request:**
```http
GET /health HTTP/1.1
Host: localhost:8001
```

**Response (200 OK):**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "2024-01-01T12:00:00.000000",
  "timestamp": "2024-01-01T12:00:00.000000"
}
```

**Response (503 Service Unavailable):**
```json
{
  "status": "degraded",
  "model_loaded": false,
  "model_version": null,
  "timestamp": "2024-01-01T12:00:00.000000"
}
```

**Поля:**
- `status`: "healthy" или "degraded"
- `model_loaded`: true если модель загружена
- `model_version`: версия модели из metadata.json
- `timestamp`: время ответа

### POST /predict

**Назначение:** Получение предсказания на основе последних свечей.

**Request:**
```http
POST /predict HTTP/1.1
Host: localhost:8001
Content-Type: application/json
```

## Формат запроса

### Структура входной свечи

Одна свеча описывается следующими полями:

| Поле | Тип | Обязательное | Описание |
|------|-----|-------------|----------|
| begin | string (ISO 8601) | Да | Время начала свечи |
| open | float | Да | Цена открытия |
| high | float | Да | Максимальная цена |
| low | float | Да | Минимальная цена |
| close | float | Да | Цена закрытия |
| volume | float | Да | Объём торгов |
| ticker | string | Нет | Ticker (default: "SBER") |
| timeframe | string | Нет | Таймфрейм (default: "1H") |

### Полный JSON Request

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
    },
    {
      "begin": "2024-01-01T11:00:00",
      "open": 250.5,
      "high": 252.0,
      "low": 249.5,
      "close": 251.0,
      "volume": 180000,
      "ticker": "SBER",
      "timeframe": "1H"
    }
  ],
  "model_version": null
}
```

### Обязательные и необязательные поля

**Обязательные:**
- `candles` — массив свечей (минимум 32 свечи для окна L=32)
- В каждой свече: `begin`, `open`, `high`, `low`, `close`, `volume`

**Необязательные:**
- `candles[].ticker` — если не указан, используется "SBER"
- `candles[].timeframe` — если не указан, используется "1H"
- `model_version` — если указан, будет использоваться (в MVP игнорируется, используется загруженная модель)

## Формат ответа

### JSON Response (200 OK)

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

### Поля ответа

| Поле | Тип | Описание |
|------|-----|----------|
| predicted_token | int | Предсказанный класс (0-6 для K=7) |
| probabilities | array[float] | Вероятности по классам (длина K) |
| action | string | Торговое действие: "buy", "sell", "hold" |
| confidence | float | Максимальная вероятность (confidence предсказания) |
| model_version | string | Версия модели из metadata.json |
| ticker | string | Ticker из последней свечи |
| timeframe | string | Таймфрейм из последней свечи |
| timestamp | string (ISO 8601) | Время предсказания |
| n_candles_used | int | Количество свечей, использованных для предсказания |
| diagnostics | object | Диагностическая информация |

### Diagnostics

| Поле | Тип | Описание |
|------|-----|----------|
| window_size | int | Размер окна (L=32) |
| n_classes | int | Количество классов (K=7) |
| horizon | int | Горизонт предсказания (h=3) |

## Коды ошибок

### 400 Bad Request

**Причины:**
- Недостаточно свечей (меньше 32 для окна L=32)
- Некорректный формат свечи (отсутствуют обязательные поля)
- Некорректный JSON

**Response:**
```json
{
  "detail": "Insufficient candles: 10 provided, need at least 32"
}
```

### 503 Service Unavailable

**Причины:**
- Артефакты модели не найдены (model.pkl, tokenizer.pkl, metadata.json)
- Модель не загружена

**Response:**
```json
{
  "detail": "Model artifacts not found: ml/artifacts/model.pkl"
}
```

### 500 Internal Server Error

**Причины:**
- Ошибка в preprocessing (некорректные данные)
- Ошибка модели
- Другие внутренние ошибки

**Response:**
```json
{
  "detail": "Prediction failed: ..."
}
```

## Типовые причины ошибок

### 1. Insufficient candles
**Причина:** API требует минимум 32 свечи для построения окна L=32.
**Решение:** Отправляйте минимум 32 последних свечи в запросе.

### 2. Invalid candle format
**Причина:** Отсутствуют обязательные поля (begin, open, high, low, close, volume).
**Решение:** Проверьте, что все поля присутствуют и имеют корректный тип.

### 3. Model artifacts not found
**Причина:** Не запущен training pipeline, артефакты не сохранены.
**Решение:** Запустите `python -m src.models.train --config-dir configs`.

### 4. Feature computation error
**Причина:** Некорректные данные (например, close < 0, volume < 0).
**Решение:** Проверьте валидность OHLCV данных перед отправкой.

## Как Go backend должен вызывать ML API

### Пример на Go

```go
package main

import (
    "bytes"
    "encoding/json"
    "net/http"
)

type Candle struct {
    Begin    string  `json:"begin"`
    Open     float64 `json:"open"`
    High     float64 `json:"high"`
    Low      float64 `json:"low"`
    Close    float64 `json:"close"`
    Volume   float64 `json:"volume"`
    Ticker   string  `json:"ticker,omitempty"`
    Timeframe string `json:"timeframe,omitempty"`
}

type PredictRequest struct {
    Candles      []Candle `json:"candles"`
    ModelVersion *string  `json:"model_version,omitempty"`
}

type PredictResponse struct {
    PredictedToken int       `json:"predicted_token"`
    Probabilities  []float64 `json:"probabilities"`
    Action         string    `json:"action"`
    Confidence     float64   `json:"confidence"`
    ModelVersion   string    `json:"model_version"`
    Ticker         string    `json:"ticker"`
    Timeframe      string    `json:"timeframe"`
    Timestamp      string    `json:"timestamp"`
    NCandlesUsed   int       `json:"n_candles_used"`
}

func Predict(candles []Candle) (*PredictResponse, error) {
    req := PredictRequest{Candles: candles}
    
    body, err := json.Marshal(req)
    if err != nil {
        return nil, err
    }
    
    resp, err := http.Post(
        "http://localhost:8001/predict",
        "application/json",
        bytes.NewBuffer(body),
    )
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()
    
    if resp.StatusCode != http.StatusOK {
        return nil, fmt.Errorf("API returned status %d", resp.StatusCode)
    }
    
    var result PredictResponse
    if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
        return nil, err
    }
    
    return &result, nil
}
```

### Рекомендации для backend

1. **Кеширование:** Кешируйте последние свечи и делайте предсказания периодически (например, раз в минуту), а не на каждый запрос.
2. **Retry logic:** Добавьте retry logic для временных ошибок (5xx).
3. **Timeout:** Установите timeout (например, 5 секунд) для запроса.
4. **Health check:** Периодически вызывайте `/health` для проверки состояния сервиса.
5. **Error handling:** Логируйте ошибки и используйте fallback стратегию (например, hold) если ML сервис недоступен.

## Shared Schema Files

API контракт согласован со следующими файлами в `shared/schemas/`:

### shared/schemas/candles_request.json

JSON Schema для валидации запроса. Содержит:
- Структуру массива свечей
- Обязательные поля каждой свечи
- Типы данных
- Описания полей

**Использование:** Backend может использовать эту схему для валидации запросов перед отправкой на ML API.

### shared/schemas/prediction_response.json

JSON Schema для валидации ответа. Содержит:
- Структуру ответа
- Типы данных
- Описания полей
- Enum для action ("buy", "sell", "hold")

**Использование:** Backend может использовать эту схему для валидации ответов от ML API.

## Правила совместимости контракта при будущих изменениях

### Принципы совместимости

1. **Backward compatibility:** Изменения API должны быть обратно совместимыми с существующими backend клиентами.
2. **Versioning:** При критических изменениях меняйте версию API (например, `/predict/v2`).
3. **Deprecation:** Помечайте устаревшие поля как deprecated в документации, но не удаляйте сразу.

### Допустимые изменения без breaking change

- **Добавление новых полей** в response (backend проигнорирует)
- **Добавление новых optional полей** в request
- **Изменение порядка полей** в JSON
- **Расширение enum** для action (добавление новых значений)

### Требуют version bump (breaking change)

- **Удаление обязательных полей** из request/response
- **Изменение типа данных** обязательных полей
- **Изменение семантики** существующих полей
- **Удаление endpoint'ов**
- **Изменение HTTP метода**

### Процесс изменений

1. Обновить `ml/src/service/schemas.py` (Pydantic)
2. Обновить `shared/schemas/*.json` (JSON Schema)
3. Обновить этот документ (`docs/api_contract.md`)
4. Обновить backend код для использования новых полей
5. Задеплоить ML сервис
6. Задеплоить backend

### Пример безопасного добавления поля

**Новый field в response:**
```json
{
  "predicted_token": 4,
  "probabilities": [...],
  "action": "buy",
  "confidence": 0.3,
  "model_version": "...",
  "ticker": "SBER",
  "timeframe": "1H",
  "timestamp": "...",
  "n_candles_used": 32,
  "diagnostics": {...},
  "risk_score": 0.15  // Новое поле
}
```

Backend, который не знает о `risk_score`, просто проигнорирует его — это backward compatible.

### Пример breaking change

**Удаление поля:**
```json
{
  // Было:
  "action": "buy",
  "confidence": 0.3,
  
  // Стало:
  "action": "buy"
  // confidence удалён
}
```

Это breaking change — требуется версия v2 endpoint'а.
