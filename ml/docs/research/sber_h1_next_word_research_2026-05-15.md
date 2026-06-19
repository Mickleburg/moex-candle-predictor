# SBER H1: календарный audit и next-word research, 2026-05-15

## Цель

Этот документ фиксирует второй audit-этап ML/NLP части:

- убрать selection leakage из research scripts;
- объяснить, почему SBER H1 содержит 24613 свечей, а не naive 24/7 count;
- добавить строгий candle accounting и calendar diagnostics;
- добавить отдельный исследовательский путь для прогнозирования следующих candle words.

Legacy `/predict` API и текущий action-classification pipeline не менялись.

## Подтверждено кодом

- Selection helpers выбирают best только по validation-side metrics.
- `clusterer` fit-ится только на train candle shapes.
- Validation/test candle words назначаются через fitted clusterer.
- Sentence/next-word samples не выпускают target за границу своего split.
- Similarity metrics используют centroids train-fitted clusterer.
- Test metrics остаются report-only и не участвуют в selection.

## Подтверждено запуском

Accounting команда:

```powershell
python ml\scripts\sber_pipeline_accounting.py --include-calendar-diagnostics --output-json data/reports/sber_h1_pipeline_accounting_20260515.json
```

Next-word команда:

```powershell
python ml\scripts\sber_next_word_research.py --output-json data/reports/sber_h1_next_word_research_20260515.json --output-csv data/reports/sber_h1_next_word_research_20260515.csv
```

## Исправление selection leakage

Selection теперь использует только validation metrics:

- `ml/scripts/sber_nlp_research.py`: validation macro-F1, затем validation accuracy, затем deterministic grid order;
- `ml/scripts/sber_hourly_research.py`: validation macro-F1, затем validation action accuracy, затем deterministic grid order.

Test metrics продолжают считаться и записываться в отчеты, но являются только финальным отчетом и не влияют на `best`.

## Calendar Audit

Наблюдаемые данные:

| Показатель | Значение |
| --- | ---: |
| First begin | 2020-01-03 09:00 |
| Last begin | 2026-05-03 18:00 |
| Raw rows | 24613 |
| Rows after cleaning | 24613 |
| Duplicate begin rows | 0 |
| Invalid OHLC rows | 0 |
| Missing OHLCV rows | 0 |
| Unique trading dates | 1676 |
| Calendar days in actual range | 2313 |
| Weekdays in actual range | 1651 |
| Saturday dates with candles | 50 |
| Sunday dates with candles | 45 |

Наивная оценка 24/7 не подходит для MOEX equity candles. Для фактического диапазона файла naive 24/7 hourly count равен 55498, а наблюдаемых свечей 24613, то есть 44.3% от 24/7. От weekday-only 24h bars наблюдаемые свечи составляют 62.1%.

Интерпретация: источник содержит бары торговых сессий, а не непрерывные wall-clock часы. Распределение часов это подтверждает:

```text
present hours: 6..23
missing hours: 0..5
```

Количество trading dates больше количества weekdays, потому что в данных есть 95 weekend dates with candles. Диагностика выводит первые 20 weekend dates и их candle counts. Timestamps трактуются как timezone-naive local exchange timestamps; timezone conversion в этом audit не выполняется.

Большинство дней содержит около 15 свечей:

| Candles per date | Значение |
| --- | ---: |
| min | 5 |
| max | 18 |
| mean | 14.69 |
| median | 15 |
| 5% quantile | 10 |
| 95% quantile | 18 |

Gap categories:

| Gap type | Count |
| --- | ---: |
| exactly 1h | 22934 |
| intraday 2-3h | 3 |
| overnight/holiday | 1253 |
| weekend/holiday | 421 |
| very large | 1 |

Единственный very large gap: `2022-02-25 23:00` -> `2022-03-24 09:00`, что согласуется с известной остановкой российского рынка. Есть только 3 intraday gaps больше 1h внутри наблюдаемого торгового дня:

- 2022-02-24: 07:00 -> 09:00;
- 2024-02-13: 13:00 -> 15:00;
- 2025-09-13: 09:00 -> 11:00.

Эти gaps видны в raw data и не создаются cleaning/split/window logic.

Raw file coverage:

- matching raw files for SBER 1H: 1;
- loaded file: `data/raw/SBER_1H_20200103T0900_20260503T1800.parquet`;
- overlaps between raw files: 0;
- gaps between raw files: 0.

В этом запуске loader не игнорирует более ранние raw files.

## Pipeline Loss Accounting

Для `window=32`, `horizon=1`:

| Stage | Count |
| --- | ---: |
| Raw loaded | 24613 |
| After cleaning | 24613 |
| Shape rows | 24613 |
| Word rows | 24613 |
| Valid action labels | 24612 |
| Train rows | 17229 |
| Val rows | 3692 |
| Test rows | 3692 |
| Train samples | 17197 |
| Val samples | 3660 |
| Test samples | 3660 |

Математическая потеря samples по каждому split:

```text
expected samples = split_len - window_size - horizon + 1
```

Для каждого split 31 свеча уходит на context warmup, а 1 свеча не может стать target, потому что ее future label вышел бы за границу split.

Все accounting checks истинны:

- cleaned `begin` sorted;
- duplicate `begin` нет;
- invalid OHLC нет;
- missing OHLCV нет;
- split ranges не пересекаются;
- shape rows равны cleaned rows;
- word rows равны cleaned rows;
- valid labels равны `len(df) - horizon`;
- invalid labels находятся только в tail;
- sentence windows aligned.

## Next-Word Forecasting

Постановка:

```text
input:  words[t-L+1], ..., words[t]
target: words[t+1], ..., words[t+K]
```

Реализованные baselines:

- `persistence`: повторяет последнее известное слово;
- `unigram`: предсказывает самый частый train future word;
- `markov1`: direct transition от последнего context word к каждому future step;
- `tfidf_logreg`: per-horizon classifiers над context word n-grams.

Важно: этот путь был первым baseline-слоем. Он еще не является полноценной language-model постановкой `P(w[t+1..t+K] | context)`, потому что часть моделей предсказывает horizons независимо.

Similarity-aware evaluation:

- exact per-horizon accuracy;
- macro-F1 per horizon;
- sequence exact match;
- top-3 accuracy, если есть probabilities;
- mean centroid distance;
- soft similarity `exp(-distance / tau)`, где `tau` равен median nonzero train-centroid distance;
- within-nearest-3 accuracy.

## Next-Word Results

Selection validation-only: validation mean accuracy, затем validation mean soft similarity, затем deterministic config order.

| Model | Context | K | Val exact@1 | Val soft | Test exact@1 | Test soft |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| markov1 | 32 | 1 | 0.2658 | 0.9245 | 0.4068 | 0.9493 |
| markov1 | 32 | 3 | 0.2657 | 0.9240 | 0.4065 | 0.9510 |
| markov1 | 16 | 1 | 0.2652 | 0.9243 | 0.4086 | 0.9495 |
| markov1 | 16 | 3 | 0.2651 | 0.9239 | 0.4083 | 0.9511 |
| persistence | 32 | 3 | 0.2264 | 0.9120 | 0.3362 | 0.9449 |
| persistence | 32 | 1 | 0.2262 | 0.9160 | 0.3366 | 0.9467 |
| unigram | 32 | 1 | 0.1915 | 0.9210 | 0.3932 | 0.9559 |
| tfidf_logreg | 16 | 1 | 0.1975 | 0.9089 | 0.3588 | 0.9505 |

Best by validation: `markov1`, `context=32`, `K=1`.

## Почему метрики слабые

- `markov1` ловит локальную инерцию последнего слова, но не моделирует устойчивую "грамматику свечей".
- Exact sequence match для нескольких будущих слов падает быстро: если token accuracy умеренная, вероятность угадать всю последовательность примерно мультипликативно уменьшается с ростом `K`.
- Candle words являются кластерами формы свечи, а не настоящими семантическими словами; похожие формы могут попадать в разные кластеры.
- Soft similarity полезна как shape-proximity diagnostic, но слишком мягкая для выбора модели: даже `unigram` может иметь высокий soft score.
- Разрыв между validation и report-only test показывает зависимость от выбранного временного сегмента; этот результат нельзя использовать для подбора.

## Интерпретация

Текущий next-word baseline показывает, что в последовательности candle words есть локальная структура, но evidence пока слабый. Нельзя делать вывод о торговой пригодности. Следующий корректный шаг - оценивать именно sequence continuation через n-gram/backoff LM, NLL/perplexity и walk-forward folds.

## Ограничения

- Baselines не являются полноценной neural language model.
- Test metrics показаны только как report-only исторический результат этого этапа.
- Soft similarity не должна быть главным selection metric.
- Trading strategy layer здесь не строился.

## Следующие шаги

1. Добавить n-gram/backoff language model для `P(w[t+1..t+K] | words[t-L+1..t])`.
2. Оценивать multi-step sequence continuation через NLL, perplexity, top-k и beam metrics.
3. Проверить разные vocabulary sizes и clusterers без test selection.
4. Использовать next-word modeling как pretext/diagnostic path, а не как готовый trading signal.
