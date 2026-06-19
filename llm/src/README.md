# LLM Source (V2 news features)

`llm_ta/` — runtime-код блока новостных признаков:

- `features.py` — ядро: загрузка расклытий из `data/news/edisclosure/`, таксономия
  `event_name → event_type`, детерминированный расчёт фич (no-lookahead по `pub_date`),
  сборка объекта `llm_analysis`. Работает без LLM/сети.
- `providers.py` — base provider + OpenAI-совместимый адаптер (Ollama/DeepSeek/Groq/OpenRouter).
  `provider_from_name("baseline")` → `None` (детерминированный путь).
- `analyzer.py` — `NewsFeatureService`: baseline + опциональное уточнение моделью с откатом.
- `validator.py` — JSON-Schema валидация + проверка no-lookahead (`published_at <= as_of`).
- `cli.py` — CLI для одного (ticker, as_of).
