# LLM Technical Analysis Block

`llm/` - scaffold будущего блока технического анализа через локальную или open-source LLM.

Блок должен принимать structured technical snapshot: ticker, timeframe, recent candle summary, indicators, support/resistance, volume и volatility. На выходе он должен возвращать JSON по `contracts/llm_analysis.schema.json`.

Ограничения:

- LLM не имеет права торговать.
- LLM не отправляет orders.
- LLM output - weak signal, который должен быть откалиброван и проверен.
- Нельзя заявлять прибыльность или гарантии.
- Реальный LLM client пока не реализован.
