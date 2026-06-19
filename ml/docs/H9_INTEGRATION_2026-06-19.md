# H9 интеграция — фид + слив + risk_manager сведены воедино — 2026-06-19

Оба делегата отчитались; ML-блок свёл всё в одну прогонку и НЕЗАВИСИМО проверил handshake.

## Что пришло от делегатов
- **LLM/news чат:** forward-фид `data/news/dividend_calendar_upcoming.csv` — 7 предстоящих
  июль-2026 событий (MTSS·ROSN·PLZL·TATN·SNGS·SBER·VTBR), board_reco ≤ record−12ТД 7/7, медиана
  запаса 39 дн. Закрыл лаг ISS (тела раскрытий с e-disclosure). Коммит c1a9a2a.
- **risk_manager чат:** комбинатор + риск-слой; контракты `sleeve_signal` + `risk_book`; book-level
  **сектор-хедж**; режимный гейт срезает гросс на 2022. Коммит 8282224.

## Интеграция на стороне ML (этот чат)
- `load_dividend_calendar()` теперь **мержит** историю (`dividends.csv`) с forward-фидом
  (`dividend_calendar_upcoming.csv`) → будущие ex-даты видны до публикации в ISS.
- `target_positions` обрабатывает БУДУЩИЕ ex-даты через торговый счётчик дней (гибрид: точные
  позиции панели для дат внутри истории, `np.busday_count` для будущих) → слив даёт ЖИВЫЕ сигналы.
- `dividend_sleeve_monitor.py` теперь читает merged-календарь (не лагающий ISS).

## Проверка handshake (независимо, ML)
**Форма сигнала.** Мой `build_sleeve_signal` (longs-only + `hedge_recommendation`) проходит их
адаптер `sleeves.py` чисто: длинные ноги → directional; своя хедж-нога не нужна (хедж на уровне
книги); `gross` берётся фолбэком. Контракты валидны (`validate_contracts.py` зелёный).

**End-to-end на ЖИВОЙ дате (2026-07-06, через `risk_manager/scripts/demo_combine_h9.py`):**
фид → слив (5 имён: MTSS/SBER/TATN/SNGS/ROSN) → нетинг → vol-target (1.5) × режимный гейт (normal,
1.0) → **сектор-хедж (MOEXFN −0.27, MOEXOG −0.47, MOEXTL −0.27)** → лимиты бьют (gross/name/sector
caps) → 8 `risk_decision`. Три нефтегаз-имени корректно свернулись в один MOEXOG-хедж.

**Backward-compat (2026-06-21 история):** слив даёт SBER/ROSN/TATN/MGNT как раньше; на 2024-06-21 через
комбинатор → ROSN/TATN long + MOEXOG-хедж, exposure_scalar 0.87. Поведение истории не изменилось.

**Live-монитор (2026-06-19):** UPCOMING — MTSS/ROSN (вход через 2 ТД, record 09-07), PLZL (через 4 ТД).
Больше не блокируется лагом ISS.

## Статус H9
| Задача | Статус |
|---|---|
| Эдж · бэктест · no-lookahead · слив · монитор · P0 (издержки/робастность/ёмкость) | ✅ |
| Фид ex-дат (LLM) | ✅ доставлен + интегрирован |
| Комбинатор risk_manager | ✅ построен + handshake проверен |
| Forward-shadow трек | ⏳ копится; теперь LIVE-способен через фид |
| Execution · автоматизация · sign-off | позже |

**Конвейер собран и работает end-to-end на живых данных.** Единственный блокер к `is_production=true`
— накопление forward-shadow (реализуется ли run-up на свежих июль-2026 ex-датах). is_production=false.

## Операционный last-mile (для прода)
- Обновлять цены (свежие свечи) к сезону — для точного inverse-vol на живых датах (сейчас при as_of
  за пределами панели берётся последняя доступная vol; для июля-2026 нужны свежие цены).
- LLM-чат обновляет фид по мере раскрытий (`edisc_fetch_bodies.py` → `build_dividend_calendar_upcoming.py`).
- execution: пер-имя `risk_decision` → лоты + дисциплина вход −12 / выход −2.
