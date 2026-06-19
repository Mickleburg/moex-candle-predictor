# SBER 1H Model Research, 2026-05-03

## Цель

Проверить реализованные в `ml/` модели на часовых свечах SBER, подобрать параметры для прогноза будущей свечи и обновить рабочие artifacts. Текущая ML-постановка проекта предсказывает не OHLC следующей свечи напрямую, а класс будущего нормализованного return/token, который затем мапится в `sell` / `hold` / `buy`.

## Данные

- Инструмент: `SBER`
- Таймфрейм: `1H`
- Запрошенный период: `2020-01-01` - `2026-05-03`
- Фактический период после загрузки: `2020-01-03 09:00:00` - `2026-05-03 18:00:00`
- Количество свечей после очистки: `24613`
- Split по времени: train `17229`, validation `3692`, test `3692`
- Raw Parquet: `data/raw/SBER_1H_20200103T0900_20260503T1800.parquet`
- JSON с полными результатами: `data/reports/sber_h1_model_research_20260503.json`

Примечание по backend: локально `go` не установлен, поэтому HTTP backend не удалось поднять. Загрузка выполнена скриптом `ml/scripts/sber_hourly_research.py`, который повторяет backend MOEX ISS contract: тот же endpoint, `interval=60`, board `TQBR`, schema `ticker/timeframe/begin/end/open/high/low/close/volume/value/source`, Parquet в `data/raw`. В backend-код внесено исправление пагинации, потому что один ISS-запрос возвращал только первые `500` свечей.

## Исправления

- `backend/internal/moex/client.go`: `FetchCandles` теперь проходит страницы MOEX через `start=0,500,...` и собирает полный batch.
- `ml/src/features/windows.py`: исправлен target alignment. Token уже содержит будущий return на горизонте `h`, поэтому target должен соответствовать последней свече окна, а не быть дополнительно сдвинутым еще на `h`.
- `ml/src/features/tokenizer.py`: исправлена ATR-нормализация target. Вместо смешения процентного return и ATR в рублях теперь используется `price_delta / ATR`.
- `ml/src/features/windows.py` и research script: исправлен leakage в Markov baseline. Markov теперь видит только прошлые реализованные токены, доступные на момент прогноза.
- `ml/src/models/baseline.py`: baseline-модели получили `save/load`; `logistic` получила median imputation и scaling для NaN после rolling-индикаторов.
- `ml/src/service/predictor.py`: загрузка artifacts стала учитывать `model_class`; HTTP inference явно не принимает Markov artifacts, так как для них нужен отдельный token-history inference path.

## Эксперимент

Проверены:

- модели: `majority`, `markov(order=1)`, `markov(order=2)`, `logistic`, `lgbm`;
- классы токенов: `K=3`, `K=5`, `K=7`;
- горизонты: `h=1`, `h=3`, `h=6`;
- окна: `L=16`, `L=32`, `L=64`;
- две сетки LightGBM.

Критерий выбора: validation `macro_f1`, затем validation action accuracy, затем test `macro_f1`. Test использовался только для финальной проверки, не для выбора.

## Лучшие Результаты

| Rank | Model | K | h | L | Val macro-F1 | Test macro-F1 | Test action acc. |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Logistic | 3 | 1 | 32 | 0.3980 | 0.3999 | 0.4033 |
| 2 | Logistic | 3 | 3 | 32 | 0.3921 | 0.3977 | 0.3983 |
| 3 | LGBM depth4/lr0.025 | 3 | 1 | 32 | 0.3895 | 0.4073 | 0.4126 |
| 4 | Logistic | 3 | 1 | 16 | 0.3887 | 0.3902 | 0.3977 |
| 5 | LGBM depth3/lr0.04 | 3 | 1 | 32 | 0.3860 | 0.4067 | 0.4137 |

Выбранная конфигурация:

```yaml
features:
  num_classes: 3
  horizon: 1
  window_size: 32

train:
  model_type: logistic
```

После обновления конфигов выполнено `python -m src.models.train --config-dir configs`; artifacts обновлены в `ml/artifacts/`.

Важно: LGBM `K=3, h=1, L=32` дал лучший test macro-F1 (`0.4073`), но не выбран как финальный, потому что test split не использовался для подбора параметров. По честному validation-критерию победил `logistic`.

## Inference Check

Проверен `CandlePredictor` на последних `96` свечах:

- predicted token: `0`
- action: `sell`
- confidence: `0.3684`
- probabilities: `[0.3684, 0.2766, 0.3550]`
- diagnostics: `K=3`, `horizon=1`, `window_size=32`

Confidence невысокий, поэтому этот сигнал нельзя трактовать как сильный торговый сигнал.

## Выводы

Для задачи классификации следующей часовой свечи лучшая честно выбранная конфигурация - `LogisticRegressionBaseline, K=3, h=1, L=32`. Она дает умеренное преимущество над majority/Markov baselines, но качество остается ограниченным: test macro-F1 около `0.400`.

Простой long/short backtest у выбранной next-candle конфигурации отрицательный (`test total_return ~= -0.740` с комиссией `0.05%`). Это значит, что модель можно считать рабочей для research/inference pipeline, но не готовой торговой стратегией. Для торговой постановки нужен отдельный критерий отбора, пороги confidence, risk layer и walk-forward проверка.
