# H9 → автономный агент на VDS — мастер-план продакшна — 2026-06-19

> Зонтичный план: как довести H9-слив от «research-конвейера, запускаемого руками» до **автономного
> торгового агента на VDS**. Подчинённые доки: гипотеза/механика — `ml/docs/H9_DIVIDEND_SLEEVE_2026-06-18.md`;
> интеграция — `ml/docs/H9_INTEGRATION_2026-06-19.md`; прежний план доводки — `docs/H9_PRODUCTION_PLAN.md`
> (его операционные строки 5-8 раскрыты здесь). КАНОН по гипотезам — `docs/RESEARCH_HYPOTHESES.md`,
> архитектура — `docs/ARCHITECTURE_V3.md`. Контексты для новых чатов — `docs/CHAT_TASK_*.md`.

## Что значит «автономный» (требования пользователя)
Агент на VDS сам: **(1)** собирает свежие данные · **(2)** мониторит ситуацию · **(3)** реагирует на
изменения · **(4)** обновляет базу знаний · **(5)** управляет портфелем. Ниже — как каждое свойство
ложится на блоки и суточный цикл.

## Суточный операционный цикл (V3, H9-first)
Триггеры по расписанию (TZ Europe/Moscow), только в торговые дни MOEX.

**EOD (после основного клиринга, ~19:05 МСК):**
| # | Шаг | Блок | Свойство |
|---|-----|------|----------|
| 1 | Докачать сегодняшние свечи 16 имён + market context (IMOEX, сектора, Brent, RGBI) | backend/data | (1) |
| 2 | Обновить дивидендный календарь с e-disclosure (новые рекомендации СД) → `dividend_calendar_upcoming.csv` + независимый no-lookahead лог | llm | (1)(4) |
| 3 | Data-integrity гейт: свежесть (есть сегодняшний бар), нет дыр/NaN; иначе HALT + алерт | backend/data | (2) |
| 4 | ML-слив: `build_sleeve_signal(as_of=today)` → лонги в пред-ex окне, inverse-vol по СВЕЖИМ vol | ml | (5) |
| 5 | Комбинатор: нетит сливы × vol-target × режимный гейт H5 × лимиты → `risk_book` + сектор-хедж | risk_manager | (5) |
| 6 | Реконсиляция: целевая книга vs текущие позиции → лимит-ордера (вход −12/выход −2), лот-округление | execution | (5) |
| 7 | Персист состояния: позиции, ордера, **P&L-атрибуция по сливам**; обновить shadow-лог РЕАЛИЗОВАННЫМ ретёрном | agent/state | (2)(4) |
| 8 | Алерт-дайджест: что входить/выходить, статус гейта, P&L, любые data-фейлы | agent/monitoring | (2) |

**Pre-open (~09:30 МСК):** проверка ночных гэпов/халтов, подтверждение/снятие лимит-ордеров,
kill-switch. Реакция (3) = режимный гейт H5 срезает гросс в шоке · halt-on-stale-data · появилась
новая ex-дата → вход · kill-switch.

## Gap-анализ: что есть vs чего нет

**ЕСТЬ (research-grade, запуск руками):** ML-слив-сервинг (`build_sleeve_signal`) + монитор + вся
H9-валидация (эдж/no-lookahead/P0) · risk_manager-комбинатор (`risk_book`) · LLM-фид дивидендов +
скрипты обновления · загрузчики (`scripts/download_candles.py`, futures), `check_data_quality.py`,
`validate_contracts.py` · контракты.

**НЕТ (productionization):**
- **Orchestrator** (`agent/`): кода нет, README устарел (описывает мёртвый V2-кросс-секшн). Нужен
  V3 суточный state-machine цикл + персист состояния (позиции, P&L-атрибуция по сливам).
- **Execution** (`execution/`): кода нет. Нужен брокер-адаптер (paper→live), дисциплина ордеров
  (вход −12/выход −2, только limit, лот-округление), реконсиляция, дубль-защита, kill-switch, аудит.
- **Backend/data** (`backend/`): кода нет. Нужен **scheduled идемпотентный** ingest (обёртка над
  существующими загрузчиками) + integrity-гейт + freshness/halt. HTTP-сервис опционален — для
  одного VDS достаточно общей файловой/SQLite базы.
- **Runtime/scheduler:** нет cron/systemd/APScheduler. Нужны EOD + pre-open триггеры.
- **Linux-окружение + деплой:** нет Dockerfile / top-level requirements / compose. Текущее окружение —
  Windows venv (py3.14); VDS — Linux. Нужно воспроизводимое Linux-окружение, секреты (брокер-ключи),
  супервизор процессов, логирование, бэкапы.
- **Мониторинг/алерты:** канала нет. Нужен Telegram-бот (или email) + health-checks + dead-man's-switch.
- **MOEX торговый календарь (праздники):** живой счётчик торговых дней RU-holiday-наивен
  (`np.busday_count`) → тайминг входа/выхода дрейфует на майских/июньских праздниках. Correctness-баг.
- **P&L-атрибуция / shadow-ГЕЙТ:** shadow-лог пишет позиции, не РЕАЛИЗОВАННЫЙ пер-событийный ретёрн →
  гейт `is_production` нечем измерить.

## H9 ML-доводка (этот чат — конкретные шаги)
Малы и принадлежат ML; финишируемы здесь. Порядок = ценность. Детали 3+4 — `ml/docs/H9_SHADOW_GATE_2026-06-19.md`.
1. ✅ **ВЫПОЛНЕНО. MOEX holiday-aware счётчик торговых дней.** `np.busday_count` в
   `dividend_sleeve.py::target_positions` (ветка будущих дат) и `dividend_sleeve_monitor.py::td_to`
   заменён на **общий `backend/trading_calendar.trading_days_between`** (backend отдал, commit 6d0c338;
   graceful fallback на np.busday_count с warning, если backend не на path). Регрессии нет (июль-2026
   без праздников → live-сигналы идентичны: smoke 19/19, sim hedged +0.526/IS +0.84, handshake 5 имён);
   майско-июньские праздники теперь корректно пропускаются (May20→Jun15: 18→17, skip 12 июня).
2. ⏳ **Свежий ценовой панель до сегодня + сверка live inverse-vol.** Ждёт автономный ingest backend-чата
   (панель сейчас по 2026-06-16). Сверка: `target_positions` на свежей панели воспроизводит размеры монитора.
3. ✅ **ВЫПОЛНЕНО. Realized-P&L shadow-гейт** — `ml/scripts/h9_shadow_pnl.py`. Методология идентична
   `runup_capture` (cross-check OK). IS-бенчмарк n=117 net **+1.24%** %pos 0.65 dose-response держится;
   FORWARD ≥2025 n=12 net **−0.93%** dose-инвертирована, не отделён от placebo. **Гейт NOT MET** →
   `is_production=false` остаётся (это и был единственный блокер). Порог `FWD_GATE_MIN_EVENTS=25`+net>0+sign-off.
4. ✅ **ВЫПОЛНЕНО. Сверка якоря (record vs ex date)** — `ml/scripts/h9_anchor_sverka.py`, **PASS 4/4**:
   оба источника якорятся на RECORD-дату (тот же объект, что в research); фид ex=record−1 ТД (T+1) 7/7;
   мерж корректен 7/7; выход −2 ТД до ex-гэпа. Слив торгует правильное событие, off-by-one нет.
5. *(опц.)* **Live serving-CLI + emit контракта.** `predict_dividend_sleeve.py` пишет `sleeve_signal`
   JSON в известный путь, чтобы `agent/` потреблял ML без импорта внутренностей. По запросу оркестратора.

## Фронт productionization → разбивка по чатам
**Одного чата НЕ хватит** — это мульти-блочный productionization, и проект уже идёт по схеме
«блок = чат» со швами на JSON-контрактах. Рекомендация — **4 новых чата + 2 ongoing**:

| Чат | Скоуп | Контекст-док | Зависит от |
|-----|-------|--------------|-----------|
| **ML (этот)** | H9-доводка 1-5; владеет слив-сервингом + shadow-P&L гейтом | — | — |
| **backend/data** *(нов.)* | автономный ingest: scheduled идемпотентная докачка свечей+контекста, integrity-гейт, freshness/halt, раскладка хранилища | `CHAT_TASK_backend_ingestion.md` | — (кормит всех) |
| **execution** *(нов.)* | брокер-адаптер paper→live, дисциплина ордеров, реконсиляция, дубль-защита, kill-switch, аудит | `CHAT_TASK_execution.md` | risk_book |
| **agent + infra** *(нов., совмещ.)* | V3 суточный цикл (ingest→signal→combine→execute→persist→alert) + state-store + scheduler + **VDS-деплой** (Linux-окружение, секреты, супервизор, логи, алерты, бэкапы) | `CHAT_TASK_agent_orchestrator.md` | всё (он клей) |
| **LLM/news** *(ongoing)* | держать дивидендный фид свежим по расписанию + no-lookahead лог; позже H8 | — (уже запущен) | — |
| **risk_manager** *(ongoing)* | live-состояние портфеля + потребление P&L-атрибуции + гейт-live | — (уже запущен) | state-store |

**Почему agent+infra вместе:** цикл и его рантайм тесно связаны (то, что деплоим = и есть цикл). Если
чат перегрузится — отделить `infra/deploy` в отдельный (контекст-док уже секционирован под это).

**Порядок/параллель:** backend/data и ML-доводка идут СРАЗУ параллельно (ни от кого не зависят).
execution — параллельно (его узкое место — выбор брокер-API, см. ниже). agent+infra собирает поверх,
поэтому стартует, как только backend отдаёт стабильную докачку и execution — paper-адаптер; но
скелет цикла и VDS-bring-up можно делать параллельно на моках.

## Путь к `is_production=true` (acceptance-гейты)
1. ML-доводка: ✅ realized-P&L гейт (3) + сверка якоря (4) + holiday-календарь (1, потребляет
   `backend/trading_calendar`); ⏳ свежий панель (2) ждёт первый EOD-ingest backend. **Гейт построен и
   сейчас NOT MET** (forward тонкий/минус) — это и есть критерий снятия `is_production`.
2. backend гонит докачку автономно ≥2 недели без дыр; integrity-гейт ловит фейлы.
3. Оркестратор гоняет полный цикл на VDS в **paper** (execution paper-режим) сезон дивидендов.
4. **Forward-shadow гейт:** realized run-up на свежих ex-датах (июль-2026+) net>0, согласуется с
   историей (+0.84%/событие, high-yield +1.45%) — измеряется шагом ML-3.
5. Paper-прогон совпал с симуляцией; алерты/health/kill-switch проверены.
6. **Sign-off команды** → снять `is_production=false`. Только после этого — включение live (явный флаг).

## Решения, требующие пользователя (не блокируют старт)
- **Брокер-API** (execution): кандидаты — T-Invest API (есть sandbox = простейший paper-путь),
  Finam TradeAPI, ALOR OpenAPI, QUIK-коннектор. Влияет на execution-чат; sandbox-first рекомендую.
- **VDS-провайдер/ОС** (infra): Linux (Ubuntu LTS). Docker vs venv+systemd — реши в infra-чате.
- **Канал алертов:** Telegram-бот (рекоменд.) vs email.
- **Капитал/лимиты live:** ёмкость слива ~130-190 млн ₽ (P0) — реальный размер задаёт лимиты.

## Дисциплина (на всех этапах, для всех чатов)
Валидация только deployment-sim на свежем forward · no-lookahead (цена ≤ as_of, новость по публикации) ·
кросс-блочный git: каждый чат коммитит **ТОЛЬКО свои файлы** · `is_production=false` до shadow-гейта +
sign-off · live запрещён без явного enable-флага (paper-first) · секреты только в `.env`/секрет-сторе,
никогда в git · `python -m pytest ml/test_smoke.py` зелёный перед PR.
