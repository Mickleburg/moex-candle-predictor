# H5 — Режимный детектор + V2-сервинг `aggregated_signal` — 2026-06-16

## H5 — детектор новизны режима

Past-only вектор режима (`src/features/regime.py`): EWMA-vol рынка, тренд IMOEX,
кросс-секц. дисперсия доходностей, среднее |return| рынка. Новизна = скользящая
**Mahalanobis**-дистанция текущего вектора к расширяющемуся прошлому распределению.
Скрипт: `ml/scripts/regime_detector_research.py`.

### Face validity — отличная
Топ-10 всплесков новизны — **все 2022**: 2022-02-24 (distance 22.9, вторжение/обвал),
2022-03..04 (санкции, ре-открытие). Детектор точно ловит структурный слом.

### Премиса: предсказуемость падает в новых режимах
Momentum IC (L20/H20) и дисперсия исходов по терцилям новизны:

| Период | low-novelty IC | high-novelty IC | disp low→high |
|--------|----------------|-----------------|---------------|
| ALL | +0.070 | +0.009 | 0.054 → 0.067 |
| IS (2020-24) | +0.070 | **−0.017** | 0.059 → 0.077 |
| FORWARD (25-26) | +0.066 | +0.088 | 0.044 → 0.041 |

In-sample (где живёт шок 2022) премиса держится **сильно**: в новых режимах IC схлопывается
и уходит в минус, разброс исходов растёт. На спокойном forward шока нет — разницы нет (логично).

### Вывод
Детектор — надёжный **шоковый гейт**: срезать/обнулять экспозицию при всплеске новизны
(в 2022 это вывело бы из периода, где любой кросс-секц. сигнал ломается, IC<0, дисперсия растёт).
Это оверлей хвостового риска для risk_manager, не тонкий фильтр альфы. Past-only, model-agnostic —
защитит и будущий новостной сигнал. API: `rolling_mahalanobis(regime_features(panel, market))`.

## V2-сервинг `aggregated_signal` (плумбинг решение→контракт)

`src/service/cross_sectional_signal.build_aggregated_signal()` превращает пер-тикерные СКОРЫ
модели в замороженный контракт `aggregated_signal` (universe, horizon, rankings[score/rank/
percentile/leg], market_neutral, is_production=false). Заменяет V1 per-ticker `model_registry`.
CLI `ml/scripts/predict_xsec_signal.py` гоняет путь end-to-end на текущих данных и **валидирует
против схемы** (пример выхода: `data/reports/aggregated_signal.json`). Скор сейчас —
плейсхолдер (моментум, не торгуемый); когда появится новостная fused-модель, меняется только
источник скора, всё вниз по течению (risk_manager) — без изменений.

## Артефакты
- `ml/src/features/regime.py`, `ml/scripts/regime_detector_research.py`
- `ml/src/service/cross_sectional_signal.py`, `ml/scripts/predict_xsec_signal.py`
