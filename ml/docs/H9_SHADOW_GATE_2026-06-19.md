# H9 ML-доводка — realized-P&L shadow-гейт + сверка якоря — 2026-06-19

ML-шаги 3 и 4 из `docs/VDS_AUTONOMOUS_PLAN.md` («H9 ML-доводка»). Оба — чистый ML, без межблочных
зависимостей. Скрипты: `ml/scripts/h9_shadow_pnl.py`, `ml/scripts/h9_anchor_sverka.py`. Отчёты:
`data/reports/h9_shadow_pnl.txt`, `data/reports/h9_anchor_sverka.txt`.

## Шаг 4 — сверка якоря (record vs ex date): ✅ PASS
Цель: убедиться, что forward-фид торгует ТОТ ЖЕ якорь, на котором мерился эдж, иначе окно входа/выхода
съедет на день и можно поймать ex-гэп. 4 проверки, все зелёные:
1. `dividends.csv` якорь = колонка `date` (ISS registryclosedate = RECORD); фид несёт `record_date`+`ex_date`.
2. Внутренняя консистентность фида: `ex_date == record_date − 1 торговый день` (T+1) — **7/7**.
3. Корректность мержа: будущие строки `load_dividend_calendar()` == `record_date` фида — **7/7**.
4. Инвариант «выход до ex-гэпа»: anchor=record, ex=record−1 ТД, выход=record−2 ТД → последний held-ретёрн
   в record−2 (= ex−1); ex-гэп в record−1 НЕ захватывается. Слив использует ТОЛЬКО `record_date`;
   `ex_date` — информационный, не торгуется.

**Вывод:** деплоимый слив якорится на ту же RECORD-дату, что и research. Off-by-one нет.

## Шаг 3 — realized-P&L shadow-гейт: построен; вердикт NOT MET (честно)
Измеримый критерий снятия `is_production=false`: реализуется ли market-adjusted run-up на ФАКТИЧЕСКИ
закрывшихся forward-событиях по ДЕПЛОЙ-правилу (вход −12 ТД / выход −2 ТД, stock−IMOEX, нетто комиссии)?
Методология идентична `runup_capture` из research (cross-check в скрипте: closed mean +0.01135 ==
ref +0.01135 → OK), так что гейт меряет тот же объект, что валидировался.

| | n | net/событие | %pos | dose-response |
|---|---|------------|------|---------------|
| **IN-SAMPLE (<2025) — бенчмарк** | 117 | **+1.24%** | 0.65 | ✅ держится (hi +1.58% vs lo +0.89%) |
| **FORWARD (≥2025) — shadow-трек** | 12 | **−0.93%** | 0.50 | ❌ **инвертирована** (hi −1.87% vs lo +0.01%) |

- Placebo (in-sample): forward gross −0.83% на **2-м перцентиле** placebo-полосы → пока НЕ отделён (тонко).
- **Forward-негатив сконцентрирован в ЭКСТРЕМАЛЬНЫХ доходностях** (special situations, не нормальный
  run-up): VTBR 35.1% net −5.14%, MTSS 18.1% net −6.21%, MGNT 11.6% net −3.88%, PLZL(апр) net −3.64%,
  ROSN net −1.66%/−3.57%. Положительные — умеренно-доходные: NVTK +7.06%, LKOH +2.19%, SNGS +1.84%,
  SBER +1.73%.
- **PENDING pipeline:** 7 событий июль-2026 (MTSS/ROSN/PLZL/TATN/SNGS/SBER/VTBR) ждут реализации.
- Live-монитор `dividend_shadow_log.csv`: копится в реальном времени по мере торговли июльских событий.

**Вердикт гейта:** `NOT MET` — forward (12 событий, −0.93%, dose инвертирована, не отделён от placebo)
НЕ подтверждает эдж. `is_production` остаётся false; live НЕ включать. Гейт сработал честно — это и был
единственный открытый блокер H9. Порог: `FWD_GATE_MIN_EVENTS=25` + net>0 + %pos>0.5 + sign-off.

## Наблюдение (ГИПОТЕЗА на shadow-период, НЕ тюнить на forward сейчас)
Forward-провал кластеризуется в доходностях >15% — это special situations (возобновление дивидендов
VTBR, спец-выплаты), которые ведут себя не как нормальный пред-ex run-up, на котором держится IS
dose-response. Возможная будущая РЕФИНация: кап доходности / исключение возобновлений/спецвыплат.
**Дисциплина:** это нельзя подгонять на тонком/сожжённом forward — только проверять на НОВЫХ
накопленных событиях (июль-2026+). Записано как гипотеза, не как изменение правила.

## Шаг 1 — holiday-aware счётчик ТД: ✅ ВЫПОЛНЕНО (backend отдал календарь)
`np.busday_count` в `dividend_sleeve.py::target_positions` (ветка будущих ex-дат) и
`dividend_sleeve_monitor.py::td_to` заменён на общий `backend.trading_calendar.trading_days_between`
(RU-holiday-aware, оверлей реального IMOEX-панеля + поддерживаемый `RU_HOLIDAYS`; commit backend 6d0c338).
Graceful fallback на np.busday_count с RuntimeWarning, если backend не на path (ML в изоляции). Регрессии
нет: июль-2026 без праздников → live-сигналы идентичны (smoke 19/19; sim hedged +0.526/IS +0.84; handshake
5 имён → сектор-хедж → 8 risk_decision; контракты валидны). Праздники теперь корректны (May20→Jun15: 18→17).

## Что осталось из ML-доводки
- Шаг 2 (свежий ценовой панель до сегодня) — ждёт первый EOD-ingest backend-чата; панель сейчас по
  2026-06-16 (backend подтвердил рассинхрон store — закроется первым прогоном ingest).
- Шаг 5 (опц. serving-CLI `predict_dividend_sleeve.py`) — по запросу оркестратора (agent-чат пока на
  paper-mock для execution; ML-сигнал зовётся через public API).
