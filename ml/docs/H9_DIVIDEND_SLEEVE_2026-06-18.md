# H9 дивидендный run-up — деплоимый слив (S3) — продукт ML-блока — 2026-06-18

Первый и единственный честный альфа-результат проекта, доведённый до деплоимого слива с живым
shadow-треком. Слив S3-смежный в мульти-стратегийной книге V3 (см. `docs/ARCHITECTURE_V3.md`).

## Что это
Дивидендный пред-ex **run-up**: цена бумаги market-adjusted дрейфует вверх перед ex-дивидендной датой
(ритейл гонится за дивидендом). Покупаем ~12 ТД до record date, выходим ~2 ТД до (перед ex-гэпом),
хеджируем рынок (beta=1 short IMOEX), сайзим inverse-vol. Направление-агностично к рынку (market-neutral).

## Доказательная база
- **Эдж (event-study + per-event):** net **+0.84%/событие** (high-yield +1.45%), %pos 62-68%.
  Робастен по 4 независимым осям: per-year (плюс 4-5/6 лет), окно входа (−15..−8 гладко плюс),
  dose-response (растёт с доходностью), **placebo** (z=+2.29, p≈0.01 — дивиденд-специфично).
  `ml/scripts/h9_dividend_research.py`.
- **Портфель (книга):** market-hedged full cum **+0.52, Sharpe +0.58**, maxDD −16%, IS Sharpe +0.84;
  сертифиц.-только книга Sharpe +0.61. `ml/scripts/h9_dividend_sleeve_sim.py`.
- **No-lookahead — СЕРТИФИЦИРОВАН** 3 линиями: закон (ФЗ-208) + нет скачка-объявления в окне +
  пер-событийные даты объявления (e-disclosure от LLM-чата) → ML независимо проверил **PASS 129/129**,
  медиана зазора 37 ТД. `ml/scripts/h9_nolookahead_verify.py`, `data/news/dividend_announcements.csv`.

## Артефакты сервинга
- `src/service/dividend_sleeve.py`: `target_positions(as_of)` (past-only, inv-vol, кап 0.34),
  `build_sleeve_signal(as_of)` → JSON `{sleeve:"s3_event", positions[long], hedge_recommendation:
  {method:"sector_index"}, gross_long, ...}`. Хедж — на уровне книги (risk_manager), сектор-метод (P0).
- `ml/scripts/dividend_sleeve_monitor.py`: **forward-shadow монитор** — берёт живой календарь дивидендов
  с MOEX ISS, находит имена в окне входа сейчас, сайзит, пишет снапшот в
  `data/reports/dividend_shadow_log.csv`. `--as-of YYYY-MM-DD` — демо/тест на прошлой дате.

## Честные границы (не приукрашиваем)
- **Единственная оговорка — тонкий/минусовой forward 2025** (12 событий, Sharpe −0.85). Это
  ограничение ДАННЫХ (дивиденды редки), не lookahead и не провал робастности (6 лет IS + 4 контроля
  держат эдж). Разрешается только накоплением forward → shadow.
- Низкочастотный сезонный эффект (record dates в осн. май–июль); абсолютная отдача скромная, но
  market-hedged и диверсифицируемая — это СЛИВ в книгу, не самостоятельная машина прибыли.
- **ISS-лаг:** `dividends.json` публикует только ПОДТВЕРЖДЁННЫЕ record dates и запаздывает. Для ранних
  forward-сигналов питать предстоящие ex-даты из e-disclosure (рекомендации СД, ~37 ТД раньше, LLM-чат),
  не только из ISS. Монитор сейчас на ISS (best-effort) + busday-прокси для счёта ТД.
- `is_production=false` до накопления forward-shadow трека + sign-off.

## Shadow-протокол
`python ml/scripts/dividend_sleeve_monitor.py` — регулярно (хотя бы еженедельно в дивидендный сезон):
печатает текущие holdings/upcoming, дописывает снапшот в `dividend_shadow_log.csv` (дедуп по as_of).
Критерий выхода из shadow: на свежих forward ex-датах run-up реализуется (net>0 после комиссии),
подтверждая IS-эдж вне 2025.

## Интеграция в risk_manager (handshake — другой блок)
`build_sleeve_signal` отдаёт целевые позиции слива с тегом `sleeve:"s3_event"`. risk_manager
(комбинатор) нетит сливы, применяет vol-targeting (H4) + режимный гейт (H5) + лимиты. Общий контракт
`aggregated_signal` сейчас ranking-формы (score/rank/percentile) и НЕ подходит календарному сливу —
расширение контракта под target-positions+`sleeve` это решение на стороне risk_manager (зафиксировано
как открытый пункт в `docs/ARCHITECTURE_V3.md`), здесь не форсируется.
