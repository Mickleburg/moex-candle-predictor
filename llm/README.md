# LLM News-Feature Block (V2)

`llm/` — извлекатель **структурированных новостных признаков** для кросс-секционной решающей
модели (ранняя фьюжн). Блок **НЕ** предсказывает buy/hold/sell и не принимает торговых решений —
он отдаёт по тикеру/моменту JSON по контракту `contracts/llm_analysis.schema.json`.

> Предыстория: позднее слияние независимых buy/hold/sell провалено (см. пивот в `docs/ARCHITECTURE_V2.md`).
> Роль блока сменилась на NLP-фичи. Источник новостей и механизм выгрузки — `llm/docs/NEWS_SOURCE_EDISCLOSURE.md`.

## Что делает блок

- По `ticker` + `as_of` собирает корпоративные раскрытия (e-disclosure, выгружены в
  `data/news/edisclosure/{TICKER}.parquet`) в окне `window_hours` назад.
- Считает признаки: `sentiment`, `impact_score`, `novelty`, `event_type`, `news_count`,
  `recency_minutes` (+ опц. `embedding`), плюс `sources[]` с `published_at` и `affected_tickers`.
- **No-lookahead по времени ПУБЛИКАЦИИ**: учитываются только раскрытия с `pub_date <= as_of`
  (никогда `event_date`). Валидатор отвергает любой источник с `published_at > as_of`.
- Детерминированный baseline работает **без LLM и без сети** (бесплатно, воспроизводимо).
  Опциональный LLM-слой уточняет `sentiment/novelty/impact/event_type` через тот же контракт.

## Что блок НЕ делает

Не торгует, не шлёт orders, не возвращает buy/hold/sell, не меняет portfolio, не вызывает
risk_manager/execution. Решение принимает кросс-секционная модель ниже по конвейеру.

## Вход / Выход

Вход — `examples/sber_request.example.json`:
```json
{ "ticker": "SBER", "as_of": "2024-06-03T18:00:00+03:00", "timeframe": "1H", "window_hours": 72 }
```
Выход — `examples/sber_llm_analysis.example.json` (валиден по `schemas/llm_analysis.schema.json`,
копия замороженного `contracts/llm_analysis.schema.json`). `is_production=false` до честного
forward-гейта.

## Запуск

```powershell
# детерминированный baseline (без LLM, бесплатно)
$env:PYTHONPATH="src"; python -m llm_ta.cli --ticker SBER --as-of 2024-06-03T18:00:00+03:00

# уточнение внутренней моделью Positive LLM (Gemma) — нужен ключ в env
$env:POSITIVE_LLM_API_KEY="<ключ>"
$env:PYTHONPATH="src"; python -m llm_ta.cli --ticker SBER --as-of 2024-06-03T18:00:00+03:00 --provider positive
```

## LLM-провайдер (опционально; не Claude — по требованию стоимости)

Слой OpenAI-совместимый: провайдер — это конфиг (`--provider` + env), не код. Учтены нюансы vLLM:
structured output через `json_schema`, **никогда `temperature=0`** для Qwen/Gemma (зацикливание),
`extra_body→top-level` (напр. `chat_template_kwargs.enable_thinking`), retry с backoff+джиттером
на 429/5xx, обход корпоративного прокси для внутреннего шлюза. При ошибке/невалидном ответе блок
откатывается к детерминированному baseline (всегда валиден).

| `--provider` | Модель | Стоимость | Прим. |
|--------------|--------|-----------|-------|
| `positive` (дефолт для LLM) | `positive-llm-chat` (Gemma-4-31B) | внутренняя, бесплатно | лучший русский, быстрый, reasoning off |
| `positive-qwen36` | `positive-llm-qwen36` (Qwen3.6-35B) | внутренняя | thinking off для классификации |
| `positive-strong` | `positive-llm-experimental` (Qwen3.5-397B) | внутренняя | reasoning-only, для сложного анализа |
| `deepseek` | `deepseek-v4-flash` | ~$0.14/$0.28 за 1M (вся вселенная <~$2) | план на перспективу; `DEEPSEEK_API_KEY` |
| `ollama` / `openai-compatible` | по `LLM_MODEL` | локально/прочее | `LLM_OPENAI_BASE_URL`, `LLM_MODEL` |
| `baseline` (дефолт) | — | бесплатно | детерминированный, без сети |

ENV для Positive LLM: `POSITIVE_LLM_API_KEY` (или `API_KEY`); базовый URL по умолчанию
`https://api-llm.ml.ptsecurity.ru/v1` (переопределяется `LLM_OPENAI_BASE_URL`).

## Smoke test

```bash
cd llm && python smoke_test.py     # baseline на реальных SBER-раскрытиях: схема + no-lookahead
```
Требуется `data/news/edisclosure/SBER.parquet`. `jsonschema` желателен (есть встроенный fallback).
