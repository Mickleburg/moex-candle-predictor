# Контекст и задача для НОВОГО ЧАТА (risk_manager блок) — комбинатор сливов + интеграция H9

> Скопируй весь файл в новый чат risk_manager-блока. Он самодостаточен.

## Контекст проекта
MOEX trading agent, ветка `change-strategy`, мульти-стратегийная архитектура V3
(`docs/ARCHITECTURE_V3.md` — прочитай). Идея: прибыльный агент = портфель слабо-коррелированных
**стратегийных сливов** под общим РИСК-СЛОЕМ. ML-блок закрыл market-neutral ранжирование/пары, но
нашёл первый робастный эдж — **дивидендный run-up слив (S3)** — и отдал его деплоимым.
`risk_manager/` сейчас скаффолд (пустой) — его надо построить как **комбинатор + риск-слой**.

## Что уже готово (входы для тебя)
- **Сигнал слива H9:** `ml/src/service/dividend_sleeve.py::build_sleeve_signal(as_of)` → JSON
  `{sleeve:"s3_event", strategy:"dividend_runup", positions:[{ticker,weight,leg:"long"}],
  hedge_recommendation:{method:"sector_index", fallback:"imoex_beta_adjusted", notional}, gross_long,
  market_neutral:true, is_production:false}`. **Только ДЛИННЫЕ ноги** (inverse-vol, кап 0.34); хедж
  слив НЕ эмитит — ты хеджируешь на УРОВНЕ КНИГИ по сектору (см. ниже). Ёмкость слива ~130-190 млн ₽
  (P0-анализ `ml/scripts/h9_capacity.py`) — учти в лимитах.
- **Риск-аналитика (H4/H5):** `ml/src/service/risk_analytics.py` + CLI `predict_risk_analytics.py` →
  контракт `contracts/risk_analytics.schema.json`: пер-тикер vol-прогноз (H4) + режимный гейт (H5,
  `exposure_scalar∈[0,1]`, срезает гросс в шоковом режиме).
- Контракты в `contracts/`: `aggregated_signal` (сейчас ranking-формы), `risk_decision`,
  `order_request`, `portfolio_snapshot`. Валидатор `scripts/validate_contracts.py`.

## Задача
Построить **risk_manager-комбинатор**, который:
1. **Принимает целевые позиции сливов** (пока один: H9 `build_sleeve_signal`; дальше — другие).
2. **Нетит** позиции по тикерам через сливы (один тикер из разных сливов → одна нетто-позиция).
3. **Применяет риск-слой:** vol-targeting (масштаб книги к целевой волатильности по H4) ×
   режимный гейт (`exposure_scalar` из H5) × **лимиты** (на имя / сектор / гросс) × кап на корреляцию.
   **Хедж H9-слива: предпочтительно ПО СЕКТОРУ** (P0-анализ ML: сектор-хедж Sharpe +0.92/DD −0.105 vs
   IMOEX beta=1 +0.54/−0.173 — run-up это эффект бумага-vs-сектор). Хеджирование — на уровне книги.
4. **Эмитит** итоговые целевые позиции (контракт `risk_decision` / `order_request`) для execution.

## Контрактная работа (обязательно, обратносовместимо)
Текущий `aggregated_signal` — кросс-секционный ranking (score/rank/percentile/leg, minItems 2) — НЕ
подходит календарному сливу (0-3 имени, target-веса, не ранги). Расширь контракт ОБРАТНОСОВМЕСТИМО
под **target-positions + поле `sleeve`** (или заведи `sleeve_signal` контракт), чтобы комбинатор
принимал и ranking-сливы (S1/S2 — закрыты, но форма), и position-сливы (S3 H9). Реши форму на своей
стороне; обнови `scripts/validate_contracts.py`.

## Приёмка
- Комбинатор потребляет `build_sleeve_signal` H9 и выдаёт валидный `risk_decision`.
- Режимный гейт реально срезает гросс при `novel=true` (проверь на 2022-кейсе).
- Лимиты применяются (ни одно имя/сектор/гросс не превышает кап).
- `is_production=false` сквозь все артефакты; валидация контрактов зелёная;
  `python -m pytest ml/test_smoke.py` не сломан (не трогай ML-тесты).

## Дисциплина
Коммить ТОЛЬКО свои файлы (`risk_manager/…`, `contracts/…` для своих изменений, свои тесты);
не трогай `ml/…`, `llm/…`, `data/…`. Источник правды — `docs/ARCHITECTURE_V3.md`,
`docs/RESEARCH_HYPOTHESES.md`. Валидация — deployment-sim; режимный гейт уже доказан (H5). Отчитайся.
