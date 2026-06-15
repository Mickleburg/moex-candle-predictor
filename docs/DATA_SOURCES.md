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

## Новости / сентимент (для Layer 3A — ИССЛЕДОВАТЬ, ещё не подключено)

Кандидаты-источники (проверить API/лицензии/покрытие):
- MOEX news / корпоративные раскрытия (e-disclosure.ru).
- РБК, Интерфакс, ТАСС, Финам — RSS/API.
- Telegram-каналы (через Telegram API) — быстрые рыночные новости.
- Агрегаторы/датасеты финансовых новостей RU.

Задача: лента → маппинг новость↔тикер → LLM-фичи (sentiment, тип события, impact, эмбеддинги).
Контракт `llm_analysis` переопределить под фичи (см. ARCHITECTURE_V2).

## Окружение

Windows venv: `ml/.venv-win` (Python 3.14). Запуск: `& "ml\.venv-win\Scripts\python.exe" ...`,
вывод в UTF-8: `$env:PYTHONIOENCODING="utf-8"`.
