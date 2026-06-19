# Backend Block (Python — to build)

`backend/` — блок данных торгового агента. **Go-реализация удалена (2026-06-15); блок переписывается
на Python** (весь стек проекта — Python).

## Назначение

- загрузка свечей MOEX (ISS) и хранение (raw parquet / БД);
- источник **market context** для V2: индексы (IMOEX/RTSI/MOEXFN/MOEXOG/RGBI), непрерывные фьючерсы
  (Brent/NG), FX-прокси — то, что ML-блок сейчас self-fetch'ит при инференсе;
- (позже) приём ленты новостей для LLM-блока (см. `docs/DATA_SOURCES.md`);
- HTTP-сервис, отдающий данные по JSON-контрактам (`contracts/`).

## Статус

Скаффолд. Пока research качает данные напрямую, минуя бэкенд:
`scripts/download_candles.py`, `scripts/download_futures_continuous.py` (оба идут в MOEX ISS).
Переиспользуемая логика для Python-бэкенда уже есть в `ml/src/service/market_context.py`
(`MarketContextProvider`) и загрузчиках в `scripts/`.

## Эндпоинты MOEX ISS и инструменты

См. `docs/DATA_SOURCES.md` — все паттерны URL, коды инструментов, нюансы tz/FX, пагинация.

## Что нужно сделать (Python-реимплементация)

1. Сервис загрузки/кэширования свечей и market context (обернуть существующие загрузчики).
2. HTTP-слой под контракты `candle_batch` / market context.
3. Гигиена tz (МСК) — как в `ml/src/data/load.py` (`to_moscow_time`, `tz_aware`).
4. Источник новостей (Layer 3A) — позже, под LLM-блок.
