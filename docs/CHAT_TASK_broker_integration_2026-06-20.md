# Задачи: backend + execution — подключение брокера T-Invest (разблокировано токеном)

> Выдано лид-чатом. Пользователь заводит брокерский счёт Tinkoff и генерит T-Invest API-токены
> (sandbox + read-only сейчас; full-access позже). Это разблокирует две отложенные задачи. Каждый чат —
> свой раздел, коммитит ТОЛЬКО свой блок. Деплой на VDS — ПОЗЖЕ; сейчас build+test локально.

## Контекст
Брокер выбран execution-чатом: **T-Invest API (Tinkoff), sandbox-first** (`execution/src/brokers/tinvest.py`).
Live дважды загейчен: `EXECUTION_ALLOW_LIVE=1` + `backend.instruments.all_verified()` (FIGI верифицированы).
Сейчас FIGI курированы, но `figi_verified=false` → live заблокирован, пока их не сверят с авторитетным
дампом инструментов T-Invest. Токен кладётся в `.env` (`TINVEST_TOKEN`), НИКОГДА в git.

---

## BACKEND — верификация FIGI против дампа T-Invest 🟠
1. Скрипт/режим (есть заготовка `scripts/build_instrument_metadata.py --tinvest-dump`): по
   read-only/sandbox токену из `.env` тянет список инструментов T-Invest (`InstrumentsService`), сверяет
   ticker↔FIGI↔lot↔ISIN с `config/instruments.json`, **флипает `figi_verified=true`** для совпавших, явно
   репортит расхождения (FIGI/лот не сошлись) — их НЕ автофиксить молча, показать для решения.
2. После сверки `backend.instruments.all_verified()` → true для 16-именной вселенной → снимает FIGI-гейт
   live (но live всё ещё за `EXECUTION_ALLOW_LIVE`).
3. (Мелочь из код-ревью) `backend/integrity.py:158` — тип-хинт `required_last: dict[str, date]` поправить
   на `dict[str, dict[str, date]]` (рантайм верен, хинт врёт).

**Приёмка:** по токену FIGI сверены, `all_verified()` отражает реальность, расхождения видны; backend-тесты
+ ml smoke зелёные; токен не в git.

---

## EXECUTION — wire-тест T-Invest sandbox 🟡
1. С `TINVEST_TOKEN` (sandbox) в `.env` и верифицированными FIGI (от backend) прогнать `TInvestBroker`
   против **sandbox**: открыть тестовый limit-ордер на 1-2 имени, подтвердить, что форма запроса/ответа
   (order_request → execution_report) совпадает с проводным API, проверить sandbox-fill, отмену, дубль-защиту.
2. Зафиксировать в README точную последовательность включения live (флаги, верифиц. FIGI, sign-off) — но
   **live НЕ включать** (нет sign-off, shadow-гейт NOT MET).
3. Подтвердить, что цены ордеров в live берутся из котировок брокера (T-Invest даёт real-time quotes по
   счёту — отдельная платная подписка на данные НЕ нужна).

**Приёмка:** sandbox-прогон исполняет тест-ордер end-to-end (формы совпали, fill/cancel/dup ок); live
по-прежнему загейчен; execution-тесты + ml smoke зелёные; токен не в git.

---

## Дисциплина
Коммить ТОЛЬКО свой блок. Секреты (`TINVEST_TOKEN`) — только в `.env`. `is_production=false` сквозь
артефакты; live за двойным флагом + sign-off. Лид перепроверит. Отчитайтесь коммитами.
