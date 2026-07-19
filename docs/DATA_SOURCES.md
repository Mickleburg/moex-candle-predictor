# Источники данных и API — справочник

## MOEX ISS (свечи, индексы, валюта, фьючерсы)

- Документация ISS: https://iss.moex.com/iss/reference/
- moexalgo (Python): https://github.com/moexalgo  ·  https://moexalgo.github.io/docs/api
- MOEXPy (примеры): https://github.com/cia76/MOEXPy
- База: `https://iss.moex.com`

### Свечи (проверенные эндпоинты, `interval`: 1, 10, 60=1H, 24=1D)
```
Акции:    /iss/engines/stock/markets/shares/boards/TQBR/securities/{SECID}/candles.json
Индексы:  /iss/engines/stock/markets/index/securities/{SECID}/candles.json        (без board)
Валюта:   /iss/engines/currency/markets/selt/securities/{SECID}/candles.json
Фьючерсы: /iss/engines/futures/markets/forts/securities/{SECID}/candles.json
Параметры: ?iss.meta=off&iss.only=candles&from=YYYY-MM-DD&till=YYYY-MM-DD&interval=60&start=N
Пагинация: по 500, увеличивать start.
```
Загрузчики: `scripts/download_candles.py` (реестр инструментов), `scripts/download_futures_continuous.py`
(непрерывный фронт-контракт для BR/NG, помесячные SECID `{ASSET}{monthcode}{yeardigit}`).

### Использованные инструменты (1H, 2020-2026, в `data/raw/`)
| SECID | Что | Драйвер для |
|-------|-----|-------------|
| SBER, GAZP, LKOH | акции (TQBR) | целевые |
| IMOEX | индекс MOEX (₽) | рынок/бета |
| RTSI | индекс RTS ($) | RTSI−IMOEX ≈ USD/RUB |
| MOEXFN | сектор финансы | SBER |
| MOEXOG | сектор нефтегаз | LKOH/GAZP |
| MOEXMM | сектор металлы | GMKN и др. |
| RGBI | индекс ОФЗ | ставки → банки |
| BR_CONT | непрерывный Brent (FORTS) | нефть → LKOH/нефтянка |
| NG_CONT | непрерывный газ (FORTS) | газ |
| CNYRUB_TOM | юань/рубль | FX (живой после 2024) |

**Нюансы:**
- USD/RUB спот (`USD000UTSTOM`) остановлен с июня 2024 → использовать RTSI−IMOEX спред или CNYRUB.
- Индексы торгуются только основную сессию (~10:00-19:00 МСК) → ~14.8k баров против 25k у акций;
  выравнивание `merge_asof(backward)`.
- Месячные коды фьючерсов: F G H J K M N Q U V X Z = янв..дек; год = последняя цифра.
- **TZ:** сырые `begin` — МСК wall-clock, исторически помечены UTC. `load_candles(tz_aware=True)` /
  `to_moscow_time()` дают корректный Europe/Moscow (час/dow не меняются). Подробно — module docstring
  в `ml/src/data/load.py`.

## Дивиденды (слив H9/S3) — ДВА РАЗНЫХ источника, не путать

| Что нужно | Источник | Даёт | Ограничение |
|---|---|---|---|
| **История** (прошлые выплаты) | ISS `/iss/securities/{SECID}/dividends.json` | `secid, isin, registryclosedate, value, currencyid` | смотрит только НАЗАД; **нет даты объявления** |
| **Опережающий календарь** (будущие отсечки) | e-disclosure.ru — рекомендации СД, тела сообщений | record/ex/`board_reco_date`/value/status | **за WAF (ServicePipe): нужен браузер** |

Файлы: `data/raw/dividends.csv` (история, ISS; фетчер `backend/dividends.py`) +
`data/news/dividend_calendar_upcoming.csv` (forward, e-disclosure; билдер
`llm/scripts/build_dividend_calendar_upcoming.py`). `load_dividend_calendar` их **склеивает** —
слив H9 видит и прошлые, и предстоящие даты.

**⛔ ISS НЕ МОЖЕТ заменить e-disclosure для forward-фида (разведка 2026-07-19, NO-GO).**
Отчёт: `llm/docs/ISS_DIVIDEND_SOURCE_RECON_2026-07-19.md`. Две независимые причины:
1. **Структурная (не чинится обновлением данных):** в схеме `dividends.json` ровно 5 полей и **ни
   одного временнóго якоря кроме `registryclosedate`**. Даты рекомендации СД / ГОСА / появления
   строки нет → no-lookahead на этом эндпоинте **непроверяем в принципе**. Взять record-дату за
   «когда мы узнали» = завести скрытую утечку прямо в H9-гейт.
2. **Эмпирическая:** эндпоинт заморожен, а не лагает — выборка 30 бумаг, **0 записей 2026**,
   глобальный максимум `registryclosedate` = 2025-08-13 (~11 мес). Не показывает даже события, чья
   record-дата прошла 13 дней назад. Контроль: соседний `capitalization.json` свежий → мёртв именно
   дивидендный сервис, а не ISS.

Дивидендного календаря / corporate-actions в бесплатном ISS **не существует**: каталог `/iss.json`
их не содержит, 6 правдоподобных путей → 404, страница moex.com рендерится JS без открытого
бэкенда. ALGOPACK — платный (не берём). НРД (`nsddata.ru`) — 403.

**e-disclosure: браузер обязателен.** Plain-`requests` получает **HTTP 403 на всё, включая корень
сайта** (глухой блок WAF, не challenge) → «переписать тот же источник на requests» невозможно.
Playwright нужен именно как обход WAF.

**Операционное следствие:** рефреш фида **не может** работать на боевом VDS (961 МБ, slim-образ —
chromium рядом с EOD-циклом = OOM). Рефреш гоняется ВНЕ VDS, на VDS доставляется готовый
провалидированный CSV; `AGENT_LLM_REFRESH_CMD` на сервере **постоянно выключен** (см. `infra/README.md`).

## Новости / сентимент

- **e-disclosure.ru** — ПОДКЛЮЧЁН, но только под дивидендные события (см. выше). Тела сообщений
  берутся через Playwright; эндпоинт тела вскрыт: `portal/event.aspx?EventId=<guid>`.
- **H2 (недельный сентимент заголовков) — ЗАКРЫТ** (нет динамического сигнала, time-shuffle ≥ real;
  см. леджер). Не переоткрывать.
- **H8 (событийные новости из тел)** — исследование в ветке `research`, отдельный лид.
- Прочие кандидаты (РБК/Интерфакс/ТАСС, Telegram-каналы) — не подключены.

## Окружение

Windows venv: `ml/.venv-win` (Python 3.14). Запуск: `& "ml\.venv-win\Scripts\python.exe" ...`,
вывод в UTF-8: `$env:PYTHONIOENCODING="utf-8"`.
