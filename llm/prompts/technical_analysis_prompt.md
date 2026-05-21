# Technical Analysis Prompt

Ты анализируешь технический snapshot российского инструмента MOEX.

Вход:

```json
{
  "ticker": "...",
  "timeframe": "...",
  "as_of": "...",
  "recent_candles_summary": {},
  "indicators": {},
  "support_resistance": {},
  "volume": {},
  "volatility": {}
}
```

Верни только JSON, совместимый с `contracts/llm_analysis.schema.json`:

```json
{
  "ticker": "SBER",
  "timeframe": "1H",
  "as_of": "2026-05-15T15:00:00+03:00",
  "technical_view": "moderately_bullish",
  "probabilities": {
    "buy": 0.42,
    "hold": 0.38,
    "sell": 0.20
  },
  "confidence": 0.55,
  "key_reasons": [
    "price above EMA20",
    "volume above average"
  ],
  "risk_notes": [
    "near resistance"
  ]
}
```

Правила:

- Не обещай прибыльность.
- Не используй формулировки гарантии.
- Не выдавай торговые приказы.
- Если данных недостаточно, снижай `confidence` и добавляй причину в `risk_notes`.
