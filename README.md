# MOEX Candle Predictor

Модульная research-платформа для торгового агента на MOEX. Блоки общаются через JSON-контракты
(`contracts/`). Весь стек — **Python**.

> **Направление V3 (2026-06-16).** Поиск ОДНОЙ модели направления/относит. силы исчерпан (см.
> леджер закрытого). Новая цель — **мульти-стратегийный агент**: портфель слабо-коррелированных
> стратегийных сливов (пары · макро-тилт · события · риск-кор) + общий риск-слой. Прибыль из
> агрегата и контроля риска. Источник правды: [`docs/ARCHITECTURE_V3.md`](docs/ARCHITECTURE_V3.md),
> [`docs/RESEARCH_HYPOTHESES.md`](docs/RESEARCH_HYPOTHESES.md) (КАНОН-леджер),
> [`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md). Рабочая ветка: `change-strategy`.

## Архитектура V3

```text
ДАННЫЕ      свечи (12+ бумаг) + market context (Brent/USDRUB/RGBI/IMOEX) + news ingestion
                          │
ПРИЗНАКИ    quant (цена/технички, кросс-секц.) ⊕ макро-беты (нефть/FX) ⊕ LLM news-события
                          │
СЛИВЫ       S1 пары-статарбитраж · S2 макро-тилт · S3 событийные новости · S4 риск-кор
                          │  ← комбинатор (нетит позиции, веса по риску)
ПОРТФЕЛЬ    risk_manager: vol-targeting (H4) + режимный гейт (H5) + лимиты + кап корреляции
                          │
ИСПОЛНЕНИЕ   execution (paper/dry-run, комиссия+проскальзывание)
```

Прибыль — из агрегата слабо-коррелированных сливов и контроля риска; отдельные сделки/сливы могут
быть в минусе. Альфа ищется в экзогенных причинах (нефть/FX/события) и бета-нейтральных спредах
(пары), НЕ в новой геометрии тех же цен (это закрыто).

## Блоки

- `ml/` — активный Python research-блок: сливы **S1 (пары)** + **S2 (макро-тилт)** + риск-аналитика (H4/H5).
- `llm/` — извлекатель новостных **событий** (S3): event_type/surprise/affected/impact из тел сообщений.
- `backend/` — данные (свечи + market context). **Python** (Go-версия удалена).
- `risk_manager/` — **комбинатор сливов** + vol-targeting + режимный гейт + лимиты.
- `execution/` — dry-run/paper/live adapter к брокеру/MOEX (пока только paper).
- `agent/` — orchestrator полного цикла.
- `contracts/` — общие JSON-схемы. `config/` — общая конфигурация.
- `aggregator/` — **удалён**: роль комбинатора ушла в risk_manager.

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
