# Backend / data block (Python)

`backend/` — фундамент данных торгового агента V3. Владеет шагами **1** (EOD-докачка) и **3**
(data-integrity гейт) суточного цикла (`docs/VDS_AUTONOMOUS_PLAN.md`) и общим **торговым
календарём MOEX**, который импортируют ML/agent. Go-реализация удалена (2026-06-15) — весь стек Python.

## Модули

| Модуль | Роль |
|--------|------|
| `backend/trading_calendar.py` | **MOEX торговый календарь, RU-holiday-aware.** Источник истины: фактические торговые дни IMOEX из панели (`data/raw`) для исторических дат + поддерживаемый список `RU_HOLIDAYS` для форварда. Drop-in замена `np.busday_count` (`trading_days_between`). Импортируется ML/agent. |
| `backend/store.py` | Файловый parquet-store: дискавери, загрузка, **идемпотентный инкрементальный мёрж** (`merge_increment`) + консолидация в один файл на `(ticker, timeframe)` (`write_consolidated`). |
| `backend/universe.py` | Вселенная ingest (16 имён + market context + фьючерсы), какие свежесть/целостность HALT-worthy (`required`). |
| `backend/ingest.py` | **Шаг 1.** Идемпотентная инкрементальная докачка (только недостающие свежие бары, ретраи/бэкофф, безопасно к повтору). |
| `backend/integrity.py` | **Шаг 3.** Гейт целостности → машинно-читаемый `OK`/`HALT` с причинами (свежесть/дыры/NaN/sync). |
| `backend/instruments.py` | Метаданные инструментов: **FIGI + round-lot + price-step** по 16 именам (для execution/agent). Источник — `config/instruments.json`. |
| `backend/api.py` | **Замороженный in-process контракт для оркестратора.** Agent импортирует ТОЛЬКО отсюда; внутренности можно рефакторить, пока держатся сигнатуры. |

## Запуск (PowerShell из корня)

```powershell
$PY = "ml\.venv-win\Scripts\python.exe"
$env:PYTHONPATH = "."

# Шаг 1 — инкрементальная докачка свечей+контекста (идемпотентно)
& $PY -m backend.ingest                 # только недостающие свежие бары
& $PY -m backend.ingest --with-futures  # + ребилд BR_CONT/NG_CONT
& $PY -m backend.ingest --backfill      # полная история для новых инструментов

# Шаг 3 — гейт целостности (читается оркестратором ПЕРЕД торговлей)
& $PY -m backend.integrity              # exit 0 = OK, 1 = HALT
& $PY -m backend.integrity --date 2026-06-16 --tolerance 1 --gap-lookback 60

# Тесты блока
& $PY -m pytest backend/tests/ -q
```

Отчёты (gitignored): `data/reports/ingest_report.json`, `data/reports/data_integrity_status.json`.

## Торговый календарь — для ML/agent

```python
from backend.trading_calendar import (
    trading_days_between,   # drop-in замена np.busday_count, но skip-ает RU-праздники
    next_trading_day, prev_trading_day, last_trading_day_on_or_before,
    add_trading_days, is_trading_day, get_calendar,
)
```

Устраняет correctness-баг: живые счётчики ТД сейчас RU-holiday-наивны (`np.busday_count` в
`ml/src/service/dividend_sleeve.py`, `ml/scripts/dividend_sleeve_monitor.py`) → тайминг входа/выхода
дрейфует на майских/июньских праздниках, где кластеризуются record-даты. Замена этих вызовов на
`trading_days_between` — задача ML-чата (этот блок отдаёт готовый общий ресурс).

`RU_HOLIDAYS` — **поддерживаемый список**; обновлять ежегодно из официального торгового календаря
MOEX / постановления о нерабочих днях. Для дат внутри панели актуальные торговые дни IMOEX
переопределяют список (ловят и ad-hoc закрытия).

## Оркестратор — стабильный in-process entry point

Agent зовёт backend живьём в процессе (без HTTP), импортируя ТОЛЬКО `backend.api` —
замороженную поверхность контракта. Сигнатуры и ключи возвращаемых словарей зафиксированы
тестом `backend/tests/test_api_contract.py` (если он падает — фасад менять осознанно).

```python
from backend import api
ingest  = api.run_ingest(with_futures=True)        # шаг 1; ingest["status"] in {ok,error}
verdict = api.check_integrity()                     # шаг 3; verdict["status"] in {OK,HALT}
if not api.is_tradeable(verdict): halt(verdict["reasons"])
if api.is_trading_day(today): enter = api.add_trading_days(record_date, -12)
figi = api.figi_for("SBER"); qty = api.round_to_lot("SBER", raw_qty)
```

`run_ingest(...) -> report` ключи: `status, reference_date, n_instruments, n_errors, n_updated, results[]`.
`check_integrity(...) -> verdict` ключи: `status, reference_date, n_fail, n_warn, reasons[], warnings[], checks[]`.

## Метаданные инструментов (FIGI / лот / шаг цены)

`config/instruments.json` — общий артефакт для execution + agent. Строится
`scripts/build_instrument_metadata.py`: **lot/min_price_step/isin** тянутся живьём из MOEX ISS
(авторитетно, без авторизации; VTBR после реверс-сплита — lot=1), **FIGI** из курируемой
T-Invest-таблицы. FIGI помечены `figi_verified=false` → ПРОВЕРИТЬ против дампа T-Invest
(sandbox-токен) перед live:

```powershell
& $PY scripts/build_instrument_metadata.py                          # ISS + curated FIGI
& $PY scripts/build_instrument_metadata.py --tinvest-dump dump.json # валидировать/перекрыть FIGI
```

`backend.instruments`: `figi_for / lot_for / round_to_lot / round_price / all_verified /
unverified_figis`. `all_verified()` — live-гейт: пока False, FIGI не подтверждены.

## Календарь: ежегодное обновление RU-праздников

`RU_HOLIDAYS` покрыт по **2027** (`RU_HOLIDAYS_THROUGH_YEAR`). Каждую осень добавлять следующий
год из официального торгового календаря MOEX (он может отличаться от федерального — MOEX иногда
торгует в «мостовой» день). `holidays_cover(date)` → False, если год за пределами покрытия, чтобы
устаревший список был ВИДИМ, а не молча мис-считал форвардные даты. Внутри панели даты
самокорректируются по фактическим торговым дням IMOEX, так что устаревание бьёт только по датам
за пределами ценовой панели.

## Решение по хранилищу (зафиксировано)

**Файловый parquet-store в `data/raw` — БЕЗ HTTP-сервиса.** Для одного VDS блоки читают общие
файлы напрямую (ML self-fetch'ит контекст через `MarketContextProvider`, читающий тот же
`data/raw`). HTTP/контрактный сервис добавил бы сетевой слой и точку отказа без выгоды на одной
машине. Идемпотентность store (`write_consolidated` = один файл на ключ) даёт безопасные
конкурентные чтения. Контрактный HTTP-слой (`candle_batch`) откладывается до мульти-хост-сценария.
`data/raw/*.parquet` регенерируемы → gitignored; трекаются код и реестры.

## Дисциплина

No-lookahead: бар в момент `t` только из `[t−window, t]`. Гейт `is_production=false` до forward
sign-off (`docs/VDS_AUTONOMOUS_PLAN.md`). Эндпоинты ISS / коды инструментов — `docs/DATA_SOURCES.md`.
Блок коммитит только свои файлы (`backend/…`); `data/` (артефакты) не коммитит.
