# MOEX Candle Predictor

Мульти-стратегийный торговый агент на MOEX. Блоки общаются через JSON-контракты (`contracts/`).
Весь стек — **Python**. Развёртывание — автономный сервис на Linux-VDS (сам собирает данные, мониторит,
управляет портфелем). Торговля по умолчанию — **paper**; live за двойным флагом.

> **Источник правды:** [`docs/ARCHITECTURE_V3.md`](docs/ARCHITECTURE_V3.md) ·
> [`docs/RESEARCH_HYPOTHESES.md`](docs/RESEARCH_HYPOTHESES.md) (КАНОН-леджер закрытого/активного) ·
> [`docs/VDS_AUTONOMOUS_PLAN.md`](docs/VDS_AUTONOMOUS_PLAN.md) (автономный деплой) ·
> [`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md). Рабочая ветка: `change-strategy`.

## Статус (2026-07-10)

**Найден первый робастный эдж — H9 «дивидендный пред-ex run-up» (слив S3).** Купить ~12 торговых дней
до record-date, выйти ~2 ТД до (перед ex-гэпом), сектор-хедж, inverse-vol сайзинг. Доказан по 4 осям
(per-year, окно входа, dose-response, placebo), no-lookahead сертифицирован, P0 пройден (издержки 4-10×
запас, робастность 24/24, ёмкость ~130-190 млн ₽). Сервинг: `ml/src/service/dividend_sleeve.py`.

**Стек задеплоен на VDS и копит forward-shadow трек (paper).** Оркестратор гоняет EOD-цикл каждый
торговый день на реальных блоках (ingest → слив → комбинатор → риск → execution-paper); Telegram-бот
мониторит по запросу (на RU-VDS Telegram идёт через прокси). `is_production=false`, 0 живого капитала —
слив держится в shadow-книге, копим сезон. Прогон блоков — 262 теста зелёные.

**Forward-гейт H9 — NOT MET** (forward тонкий/отрицательный, n=12). Ближайшее чтение знака — после
реализации июльской див-волны (~2026-07-24…27); формальный MET (n≥25) — не раньше осени 2026.

**Закрыто ⛔ (не переоткрывать):** направление 1H одной бумаги (V1); кросс-секционное ранжирование —
цена (H1), новости-заголовки (H2), макро-тилт (H6); попарная реверсия (H7). Робастная market-neutral
альфа на этой вселенной испробованными методами недоступна → сливы S1/S2 отложены. Тест 2025-2026 сожжён.

## Архитектура V3

```text
ДАННЫЕ      свечи (16 бумаг) + market context (Brent/RGBI/IMOEX/сектора) + дивидендный календарь
                          │  backend: идемпотентный ingest + integrity-гейт + RU-holiday календарь
ПРИЗНАКИ    событийный календарь (H9) ⊕ риск-аналитика (H4 vol / H5 режим)
                          │
СЛИВЫ       S3 событийный (H9 дивиденды) ✅ активен · S1 пары / S2 макро — закрыты · S4 риск-кор
                          │  ← risk_manager: нетит сливы, shadow-гейт (неподтверждённый слив → 0 риска)
ПОРТФЕЛЬ    vol-targeting (H4) × режимный гейт (H5) × лимиты × сектор-хедж × кап корреляции
                          │
ИСПОЛНЕНИЕ   execution (paper/dry-run → live за флагом): дисциплина −12/−2, лоты, kill-switch, аудит
                          │
ОРКЕСТРАТОР  agent: суточный цикл (EOD+pre-open) + SQLite-состояние + планировщик + алерты
```

**Shadow-гейт (инварианты #9/#4):** слив получает живой капитал только если `is_production=true` И
forward-гейт MET. Иначе — shadow-книга (0 живого капитала), папер-трек для атрибуции. H9 сегодня
(`is_production=false`) → shadow: оркестратор ставит **0 live-ордеров**, копит forward-трек.

## Блоки

| Блок | Роль | Статус |
|------|------|--------|
| `backend/` | данные: ingest свечей+контекста, integrity-гейт, торговый календарь, метаданные инструментов | ✅ автономный |
| `ml/` | сервинг слива H9 (S3) + риск-аналитика (H4/H5); research закрыт по market-neutral | ✅ H9 готов |
| `llm/` | самообновляемый дивидендный фид (события из тел e-disclosure, no-lookahead) | ✅ EOD-рефреш |
| `risk_manager/` | комбинатор сливов + риск-слой + **shadow-гейт** + сектор-хедж | ✅ |
| `execution/` | брокер-адаптер (T-Invest), дисциплина −12/−2, kill-switch, аудит | ✅ paper (live загейчен) |
| `agent/` | оркестратор суточного цикла + состояние + планировщик + алерты | ✅ |
| `bot/` | Telegram-мониторинг (read-only): статус/позиции/P&L/гейт, allowlist-доступ; на RU-VDS через прокси | ✅ задеплоен |
| `infra/` | Docker/compose + systemd + бэкапы для VDS | ✅ |
| `contracts/` · `config/` | общие JSON-схемы · конфигурация | — |

## Автономный суточный цикл (на VDS)

EOD (~19:05 МСК, торговые дни): ingest свечей → integrity-гейт (HALT ⇒ не торговать) → LLM-рефреш фида →
ML-слив → комбинатор+риск+shadow-гейт → execution (paper) → персист состояния+P&L → алерт-дайджест.
Pre-open (~09:30): ночные гэпы/халты, подтверждение ордеров, kill-switch. Деплой — `docs/VDS_AUTONOMOUS_PLAN.md`,
`infra/`. **Данные на VDS регенерируются сами** (первый ingest добивает историю свечей; первый LLM-рефреш
фетчит тела e-disclosure) — в гите только код/генераторы и дистиллированные срезы (см. ниже).

## Метод валидации

Только **deployment-симуляция** на свежем forward, с комиссией (обычный walk-forward обманул в V1).
Тест-сплит 2025-2026 сожжён. Forward-shadow гейт измеряет реализованный run-up на свежих ex-датах
(`ml/scripts/h9_shadow_pnl.py`) — сейчас **NOT MET** (forward тонкий/минус) → копим сезон. Робастность важнее пика.

## Safety / инварианты

- Реальная торговля запрещена по умолчанию; первый режим — dry-run/paper; live за двойным флагом.
- risk_manager сильнее всех: shadow-гейт даёт неподтверждённому сливу 0 живого капитала.
- Тест-сплит нельзя использовать для тюнинга; `is_production=false` до forward-гейта + sign-off.
- Весь JSON I/O валидируется схемами `contracts/`; каждый блок коммитит только свои файлы.

## Проверки

```powershell
$PY = "ml\.venv-win\Scripts\python.exe"
& $PY -m pytest ml/test_smoke.py agent/tests execution/tests backend/tests risk_manager/tests bot/tests   # тесты блоков (262)
& $PY scripts/validate_contracts.py                                                              # контракты
& $PY -m agent.src.cli run-eod --force                                                           # один суточный цикл (mock-дефолт, безопасно)
```

## Документация

- Архитектура V3: [`docs/ARCHITECTURE_V3.md`](docs/ARCHITECTURE_V3.md) · леджер гипотез: [`docs/RESEARCH_HYPOTHESES.md`](docs/RESEARCH_HYPOTHESES.md)
- Автономный деплой: [`docs/VDS_AUTONOMOUS_PLAN.md`](docs/VDS_AUTONOMOUS_PLAN.md) · аудит интеграции: [`docs/INTEGRATION_AUDIT_2026-06-20.md`](docs/INTEGRATION_AUDIT_2026-06-20.md)
- H9 слив: [`ml/docs/H9_DIVIDEND_SLEEVE_2026-06-18.md`](ml/docs/H9_DIVIDEND_SLEEVE_2026-06-18.md) · источники данных: [`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md)
- Блоки: `*/README.md` · история ML-research: `ml/docs/research/`
