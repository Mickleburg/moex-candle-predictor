# Follow-up задачи по чатам — после первой волны автономного стека — 2026-06-20

> Все блоки доставили первую волну (backend `6d0c338`, execution `e63eeec`, agent+infra `eabf87d`,
> ML `aa3d62f`, LLM-фид + risk_manager-комбинатор раньше). Ниже — ОСТАВШАЯСЯ работа, чтобы свести
> стек в реально автономный на VDS. Каждый блок — отдельный раздел: скопируй свой раздел в свой чат
> (контекст у чатов уже есть; это continuation, не cold-start). Зонтичный план — `docs/VDS_AUTONOMOUS_PLAN.md`.
> Дисциплина прежняя: коммить ТОЛЬКО свои файлы; live за явным флагом; `is_production=false` до shadow-гейта
> + sign-off; `python -m pytest ml/test_smoke.py` зелёный.

## ⬆ ОБНОВЛЕНИЕ ПОСЛЕ АУДИТА (2026-06-20, `docs/INTEGRATION_AUDIT_2026-06-20.md`)
Вторая волна доставлена параллельно; интеграция ЦЕЛА (206 тестов зелёные на HEAD, оркестратор гоняет
end-to-end live sleeve→combiner→execution). Закрыто из списка ниже: backend-1 (ingest, store до 06-19),
backend-2/3/4 (метаданные, api-seam, RU_HOLIDAYS), execution-1/2/3 (serve-CLI, backend-календарь,
lot/FIGI), agent-1/2/3/4 (execution off mock, единый календарь, LLM-рефреш в EOD, ML-CLI-шов), LLM-1/2
(refresh CLI + сезон), **ML 1/2/5 + 3/4 (вся ML-сторона H9 закрыта)**. ОСТАЁТСЯ (приоритет):
- **risk_manager (ТОП, корректность):** гейт слива по shadow-статусу — H9 при `is_production=false`/shadow
  NOT MET должен идти в книгу с НУЛЕВЫМ живым риском (инвариант #9/#4). Сейчас комбинатор даёт полный риск.
- **agent:** (a) «paper»-профиль конфига — все блоки live (абс. пути интерпретатора! относительный падает
  на Windows), `llm.refresh_cmd` задан, backend live; mock — дефолт для тестов. (b) мигрировать импорты на
  замороженный `backend.api`. (c) выбрать канонический шов execution (in-process vs serve-CLI).
- **execution:** sandbox wire-тест T-Invest (нужны `TINVEST_TOKEN` + верифицированные FIGI) — перед live.
- **LLM:** сдвинуть `FETCH_FLOOR` на следующий сезон (мелочь).
- **backend:** валидация FIGI против дампа T-Invest перед live (live уже загейчен `all_verified()`).
- **ML:** H9 закрыт; остаётся сезонное накопление shadow-трека. Опц. гигиена: импорт через `backend.api`.

Ниже — исходный полный список (исторический контекст).

Главные кросс-блочные швы, которые надо закрыть (ниже расписаны по владельцам):
- **execution ↔ orchestrator не сведены** (agent на paper-mock; нужен стабильный CLI execution).
- **3 дубля trading_calendar** (backend — канон; agent и execution должны потреблять его, не свой).
- **store рассинхронен/устарел** (первый backend-ingest закроет; разблокирует ML-шаг 2).
- **LLM-рефреш не на расписании** (нужен единый entry point для EOD-шага оркестратора).
- **risk_manager не гейтит слив по shadow-статусу** (H9 сейчас NOT MET → должен быть shadow-only).

---

## 1. backend/data (continue from `6d0c338`)
1. **Первый автономный EOD-ingest → освежить store до сегодня.** Ты сам нашёл рассинхрон (1H/SBER до
   2026-06-01 vs H7-имена до 2026-06-16) и устаревание. Прогони инкрементальный ingest, доведи ВСЕ
   16 имён + market context до текущей даты, подтверди, что integrity-гейт переключился OK (был HALT
   по делу). Это разблокирует ML-шаг 2 (live inverse-vol на свежих ценах) и весь живой цикл.
2. **Метаданные инструментов для execution:** FIGI-карта + round-lot размеры по 16-именной вселенной
   (T-Invest требует FIGI; у execution сейчас плейсхолдер-дефолты). Отдай одним модулем/JSON, который
   импортируют execution и agent. Источник — MOEX ISS / T-Invest instruments (см. `docs/DATA_SOURCES.md`).
3. **Стабильный entry point для оркестратора:** подтверди сигнатуру ingest+integrity+calendar, которую
   agent зовёт in-process (он пишет «backend wired live in-process»). Зафиксируй контракт, чтобы не плыл.
4. *(опц.)* Заметка о ежегодном обновлении `RU_HOLIDAYS` (2028+) из официального календаря MOEX.

**Приёмка:** ingest довёл store до сегодня; integrity OK; FIGI/lot-карта отдана; backend-тесты + ml smoke зелёные.

---

## 2. execution (continue from `e63eeec`)
1. **Стабильный CLI entry point для оркестратора.** Вход = `risk_book` JSON (путь), выход =
   `execution_report` (путь/stdout). Зафиксируй точную команду в README, чтобы agent прописал
   `blocks.execution.command` и снял execution с paper-mock. (Зеркало того, как ML/risk_manager зовутся.)
2. **Потреблять общий календарь backend.** Внедри `backend.trading_calendar` (инжектни его RU-праздники
   в `execution/src/trading_calendar.py`) — чтобы дисциплина −12/−2 считалась на ТОМ ЖЕ каноне, что слив
   и монитор. Сейчас риск дрейфа из-за своего списка праздников. Ты уже сделал календарь инъекцией —
   осталось подать backend-канон по умолчанию.
3. **Заменить плейсхолдер lot/FIGI на метаданные backend** (зависит от backend-задачи 2).
4. *(позже)* Wire-тест `TInvestBroker` против sandbox, когда появятся `TINVEST_TOKEN` (.env) + FIGI-карта.

**Приёмка:** документированная CLI-команда отдаёт `execution_report` из примера `risk_book`; календарь =
backend-канон; execution-тесты + ml smoke зелёные; live по-прежнему за флагом.

---

## 3. agent + infra (continue from `eabf87d`)
1. **Снять execution с paper-mock на реальный paper-broker:** пропиши `blocks.execution.command` в
   `agent_config.json`, как только execution подтвердит CLI (зависит от execution-задачи 1).
2. **Единый календарь:** потребляй `backend.trading_calendar` как канон — убери/замени
   `agent/src/trading_calendar.py` (или сделай его ре-экспортом backend), чтобы trading-day-гейтинг
   оркестратора совпадал со сливом и execution. (3 дубля → 1 источник.)
3. **Вписать LLM-рефреш фида в EOD-шаг 2:** оркестратор зовёт LLM refresh-CLI ПЕРЕД шагом слива, чтобы
   база знаний (предстоящие ex-даты) самообновлялась (зависит от LLM-задачи 1).
4. **ML-слив через новый CLI-шов (опц., чище):** ML отдал серверный CLI —
   `python ml/scripts/predict_dividend_sleeve.py --as-of <date> --out <path>` → валидный `sleeve_signal`
   JSON (commit ML). Можно звать его сабпроцессом вместо in-process импорта pandas/numpy — это держит
   stdlib-only ядро оркестратора чистым (как ты и хотел с execution). `--out -` отдаёт JSON в stdout.
5. *(опц., low-pri)* Миграция `validate_contracts.py` с устаревшего jsonschema `RefResolver` на `referencing`.

**Приёмка:** полный цикл гоняет реальный execution-paper (не mock); один календарь; LLM-рефреш в EOD;
agent-тесты + ml smoke + контракты зелёные; `is_production=false`, live двойно-загейчен.

---

## 4. LLM/news (continue from dividend-feed работы)
1. **Единый scheduled refresh entry point (CLI),** который оркестратор зовёт на EOD: `edisc_fetch_bodies.py`
   → `build_dividend_calendar_upcoming.py` → обновлённый `data/news/dividend_calendar_upcoming.csv` +
   повторный независимый no-lookahead чек. Идемпотентный, устойчив к сетевым сбоям. Это делает дивидендную
   базу знаний самообновляемой на VDS (свойство (4) автономности).
2. **Держать фид свежим весь сезон:** по мере новых рекомендаций СД — добавлять; отказы от дивидендов —
   убирать; инвариант `board_reco_date ≤ record − 12 ТД` (no-lookahead) держать; колонки confidence/source
   сохранять. ML независимо верифицирует якорь (`ml/scripts/h9_anchor_sverka.py`) — фид должен его проходить.
3. *(отложено, НЕ сейчас)* H8 событийные новости — после прод-H9 (память `future_hypotheses`).

**Приёмка:** одна команда обновляет фид end-to-end и проходит no-lookahead; идемпотентна; коммить ТОЛЬКО
свои файлы; CSV gitignored → `git add -f` для трекаемого среза, если нужно.

---

## 5. risk_manager (continue from комбинатора)
1. **Стабильный entry point комбинатора для оркестратора:** вход = `sleeve_signal` JSON(ы), выход =
   `risk_book`. Подтверди замороженную сигнатуру/контракт (agent зовёт in-process). Опц. — CLI-шов как у ML.
2. **Гейт слива по shadow-статусу (инвариант #9 + #4) — ВАЖНО.** Комбинатор обязан смотреть на
   `is_production` слива И на его forward-P&L атрибуцию (из state-store agent) и давать сливу без
   подтверждённого эджа **shadow-вес (0 живого капитала)**, а не реальный риск. H9 сейчас:
   `is_production=false`, shadow-гейт **NOT MET** (`ml/scripts/h9_shadow_pnl.py`: forward n=12, net −0.93%,
   dose-инвертирована). Значит H9 в книге должен быть **paper/shadow-only** до MET-гейта + sign-off.
   Сейчас комбинатор считает слив боевым — это надо изменить: уважать shadow-статус.
3. *(готовность на будущее)* Кап корреляции по `sleeve`-id — когда подключатся S1/S2/S4.

**Приёмка:** комбинатор даёт H9 shadow-вес (0 live) пока гейт NOT MET; уважает `is_production=false`;
P&L-атрибуция по сливам потребляется; risk_manager-тесты + ml smoke + контракты зелёные.

---

## Что сделал ЭТОТ чат (ML) в этой волне
- Шаг 1: holiday-aware календарь вшит в слив+монитор (`aa3d62f`).
- Шаг 5: серверный CLI `ml/scripts/predict_dividend_sleeve.py` → валидный `sleeve_signal` JSON (CLI-шов
  для оркестратора). Остаётся ML-шаг 2 (свежий панель) — ждёт backend-задачу 1 (первый ingest).
