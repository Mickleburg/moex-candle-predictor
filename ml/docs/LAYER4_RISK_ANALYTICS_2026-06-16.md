# ML-блок V2 — пивот в Layer-4: риск-аналитика (2026-06-16)

## Почему пивот
Кросс-секционная НАПРАВЛЕННАЯ альфа не нашлась — закрыто строго и независимо:
- Направленное 1H одной бумаги (V1) — провалено.
- Price-only кросс-секция (H1a/H1b) — нет переносимой альфы (3 таргета + deployment-гейт).
- Новости кросс-секция (H2): baseline-метаданные = статический байас; реальный LLM-контент
  (заголовки, недельный прогон) — нет динамического сигнала (time-shuffle), sentiment
  forward IC отрицательный/нестабильный.

Честный исход (Layer-4 из PATHS_FORWARD): ML-блок — не альфа-генератор, а **риск-аналитика**.

## Роль ML-блока теперь
Выдаёт `risk_analytics` (контракт `contracts/risk_analytics.schema.json`) для risk_manager:
- **vol_forecast** (H4): EWMA-прогноз forward-волатильности на тикер. Подтверждён — forward
  corr 0.44–0.52, QLIKE < naive (`vol_predictability_2026-06-16.md`).
- **inv_vol_weight** (H4): сайзинг 1/vol, нормированный (vol-targeting).
- **regime** (H5): Mahalanobis-новизна режима + `exposure_scalar` (срезать экспозицию в новых
  режимах). Детектор точно ловит шоки (2022), оверлей режет просадку
  (`regime_detector_2026-06-16.md`, `xsec_risk_overlay_2026-06-16.md`).

Это ИНФОРМАЦИЯ для risk_manager (сайзинг + гейтинг), не торговое решение. `is_production=false`.

## Реализация
- Контракт: `contracts/risk_analytics.schema.json` (+ пример, валидируется).
- Сервинг: `ml/src/service/risk_analytics.py` (`build_risk_analytics`).
- CLI: `ml/scripts/predict_risk_analytics.py` → схема-валидный payload на текущих данных.
- Компоненты: `ml/src/features/regime.py`, `ml/scripts/vol_predictability_research.py`,
  `ml/scripts/xsec_risk_overlay_sim.py`.

## Что дальше (Layer-4 дорожная карта)
1. **Shadow/paper:** гонять risk_analytics на свежем forward, сверять vol-прогноз и режим-флаги
   с реализацией (deployment-стиль).
2. **risk_manager интеграция:** портфельный сайзинг по inv_vol_weight + гейт по exposure_scalar.
3. (Опц.) обогащение режима (корреляционная структура, ликвидность), мульти-горизонт vol.
4. Альфа-линия остаётся открытой как РЕЗЕРВ: тела сообщений (богаче заголовков), событийная
   постановка, дневной горизонт — если будет решение вернуться.

## Артефакты-вердикты
`xsec_h1_baseline`, `xsec_target_and_gate`, `h2_baseline_news`, `h2_llm_weekly_verdict`,
`vol_predictability`, `regime_detector`, `xsec_risk_overlay` (все в `ml/docs/research/`).
