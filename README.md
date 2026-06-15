# MOEX Candle Predictor

Модульная research-платформа для торгового агента на MOEX. Блоки общаются через JSON-контракты
(`contracts/`). Весь стек — **Python**.

> **Направление V2 (2026-06-15).** Направленное предсказание 1H одной бумаги (buy/hold/sell)
> провалено и закрыто. Новая цель — **кросс-секция / маркет-нейтрал + ранняя ML/LLM фьюжн
> (LLM = новостные фичи) + новости**. Источник правды: [`docs/ARCHITECTURE_V2.md`](docs/ARCHITECTURE_V2.md),
> [`docs/RESEARCH_HYPOTHESES.md`](docs/RESEARCH_HYPOTHESES.md), [`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md).
> Рабочая ветка: `demo`.

## Архитектура V2

```text
ДАННЫЕ      backend (свечи) + market context (индексы/фьючи/FX) + news ingestion
                          │
ПРИЗНАКИ    quant features (цена/технички, кросс-секц.) ⊕ LLM news features   ← ранняя фьюжн
                          │
РЕШЕНИЕ     ml: кросс-секционная модель → ранг по относительной силе
                          │
ПОРТФЕЛЬ    risk_manager: long top-k / short bottom-k, маркет-нейтрал, лимиты
                          │
ИСПОЛНЕНИЕ   execution
```

## Блоки

- `ml/` — активный Python research-блок. Сейчас разворачивается в **кросс-секционную** модель.
- `llm/` — извлекатель **новостных признаков** (NLP), не buy/hold/sell. Текущий код — мок под
  старую late-fusion-схему, ждёт переопределения контракта.
- `backend/` — данные (свечи + market context). **Переписывается на Python** (Go-версия удалена).
- `risk_manager/` — станет **портфельным** слоем (long/short, маркет-нейтрал, лимиты, сайзинг).
- `execution/` — dry-run/paper/live adapter к брокеру/MOEX (пока только paper).
- `agent/` — orchestrator полного цикла.
- `contracts/` — общие JSON-схемы. `config/` — общая конфигурация.
- `aggregator/` — **удалён**: late fusion заменён ранней фьюжн внутри решающей модели.

## Метод валидации

Только **deployment-симуляция** (скользящий ретрейн сквозь свежий forward-период, с комиссией).
Обычный walk-forward обманул нас в V1. Тест-сплит 2025-2026 сожжён — для честного гейта нужен
свежий forward. Подробности — в `docs/RESEARCH_HYPOTHESES.md`.

## Safety

- Реальная торговля по умолчанию запрещена; первый режим — dry-run/paper.
- Risk manager сильнее всех: может заблокировать любой сигнал.
- LLM не торгует и не принимает финальное решение (в V2 он вообще выдаёт только фичи).
- Тест-сплит нельзя использовать для тюнинга.
- `is_production=false` до честного forward-гейта + sign-off.

## Проверки

```powershell
$PYTHON = "ml\.venv-win\Scripts\python.exe"
& $PYTHON -m pytest ml/test_smoke.py          # смоук-тесты ML
& $PYTHON scripts/validate_contracts.py        # валидация контрактов
```

## Документация

- Архитектура V2: `docs/ARCHITECTURE_V2.md`
- Гипотезы исследования: `docs/RESEARCH_HYPOTHESES.md`
- Источники данных (ISS/новости): `docs/DATA_SOURCES.md`
- ML-блок: `ml/README.md`, `ml/CLAUDE.md`
- LLM-блок: `llm/README.md`, `llm/CLAUDE.md`
- Контракты: `contracts/README.md`
- История ML-research: `ml/docs/research/`
