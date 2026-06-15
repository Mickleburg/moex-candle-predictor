# LLM Technical Analysis Block

`llm/` - независимый блок технического анализа через mock или локальную/open-source LLM.

Блок принимает technical snapshot по тикеру и возвращает строго структурированный JSON по `schemas/llm_analysis.schema.json`.

## Что делает блок

- Интерпретирует технический snapshot: цена, EMA, RSI, ATR, объем, support/resistance, trend/volatility regime.
- Возвращает `technical_view`, вероятности `buy/hold/sell`, `confidence`, `key_reasons` и `risk_notes`.
- Валидирует любой ответ провайдера как строгий JSON и затем по JSON Schema.
- При ошибке провайдера или невалидном JSON возвращает безопасный neutral fallback.

## Что блок НЕ делает

- Не торгует.
- Не отправляет orders.
- Не принимает финальное решение BUY/SELL/HOLD.
- Не меняет portfolio state.
- Не вызывает `aggregator`, `risk_manager` или `execution`.

Финальное действие принимает только дальнейшая цепочка: `aggregator -> risk_manager -> execution`.

## Вход

Минимальный формат:

```json
{
  "ticker": "SBER",
  "timeframe": "1H",
  "as_of": "2026-05-15T15:00:00+03:00",
  "technical_snapshot": {
    "last_close": 301.5,
    "ema_20": 300.8,
    "trend_regime": "up"
  }
}
```

Полный пример лежит в `examples/sber_1h_input.json`.

## Выход

```json
{
  "ticker": "SBER",
  "timeframe": "1H",
  "as_of": "2026-05-15T15:00:00+03:00",
  "technical_view": "moderately_bullish",
  "probabilities": {
    "buy": 0.42,
    "hold": 0.38,
    "sell": 0.2
  },
  "confidence": 0.55,
  "key_reasons": [
    "trend regime is up",
    "price above EMA20",
    "volume above average"
  ],
  "risk_notes": [
    "near resistance"
  ]
}
```

Схема требует `probabilities.buy`, `probabilities.hold`, `probabilities.sell` в диапазоне от `0` до `1`. Runtime-валидатор дополнительно проверяет, что сумма примерно равна `1.0` с tolerance `0.02`.

## Fallback

Если LLM вернула текст вместо JSON, битый JSON, неверную схему или не ответила, наружу отдается безопасный fallback:

```json
{
  "technical_view": "neutral",
  "probabilities": {
    "buy": 0.0,
    "hold": 1.0,
    "sell": 0.0
  },
  "confidence": 0.0,
  "key_reasons": ["LLM output was invalid"],
  "risk_notes": ["fallback response used"]
}
```

`ticker`, `timeframe` и `as_of` сохраняются из входа.

## Mock mode

Mock mode работает без реальной LLM:

- `trend_regime == "up"` и `last_close > ema_20` -> `moderately_bullish`;
- `trend_regime == "down"` и `last_close < ema_20` -> `moderately_bearish`;
- иначе -> `neutral`.

Запуск CLI из папки `llm/`:

```bash
PYTHONPATH=src python -m llm_ta.cli --provider mock --input examples/sber_1h_input.json
```

## Smoke test

```bash
cd llm
python smoke_test.py
```

Smoke test читает `examples/sber_1h_input.json`, запускает mock provider, валидирует результат и сравнивает его с `examples/sber_1h_expected_output.json`.

`jsonschema` не обязателен для smoke test: если пакет не установлен, используется встроенная минимальная проверка контракта. Для полноценной JSON Schema validation:

```bash
cd llm
python -m pip install -r requirements.txt
python smoke_test.py
```

## Локальный LLM provider

Реальный вызов модели отделен от бизнес-логики через provider interface.

Поддержан OpenAI-compatible local endpoint, например vLLM, LM Studio или Ollama `/v1`:

```bash
export LLM_OPENAI_BASE_URL=http://127.0.0.1:11434/v1
export LLM_OPENAI_API_KEY=ollama
export LLM_MODEL=llama3.1
export LLM_TIMEOUT_SECONDS=30

PYTHONPATH=src python -m llm_ta.cli \
  --provider openai-compatible \
  --input examples/sber_1h_input.json
```

Если local provider вернет нестрогий JSON, CLI напечатает neutral fallback.
