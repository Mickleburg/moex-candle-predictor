# SBER H1 Calendar Audit And Next-Word Research, 2026-05-15

## Scope

This note documents the second ML/NLP audit step:

- remove test-based tie-breakers from model selection;
- explain why SBER H1 has 24613 candles, not a naive 24/7 count;
- add strict candle accounting and calendar diagnostics;
- add a separate next-candle-word forecasting research path.

The existing action-classification pipeline and legacy `/predict` API remain compatible.

## Selection Leakage Fix

Selection now uses validation-side metrics only.

- `ml/scripts/sber_nlp_research.py`: validation macro-F1, then validation accuracy, then deterministic grid order.
- `ml/scripts/sber_hourly_research.py`: validation macro-F1, then validation action accuracy, then deterministic grid order.

Test metrics are still computed and written to reports, but they are report-only and do not affect `best`.

## Calendar Audit

Status: confirmed by command run.

Command:

```powershell
python ml\scripts\sber_pipeline_accounting.py --include-calendar-diagnostics --output-json data/reports/sber_h1_pipeline_accounting_20260515.json
```

Observed data:

| Item | Value |
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

The user's rough 46680 estimate assumes a shorter 24/7 span. For the actual file range,
the naive 24/7 hourly count is 55498. Observed candles are 24613, or 44.3% of 24/7.
Against weekday-only 24h bars, observed candles are 62.1%.

This is expected for MOEX shares data: the source contains trading-session bars, not
continuous wall-clock hours. The hour distribution confirms this:

```text
present hours: 6..23
missing hours: 0..5
```

The number of unique trading dates is greater than the number of weekdays because the
source includes 95 weekend dates with candles. The report lists the first 20 weekend
dates and their candle counts. Timestamps are treated as timezone-naive local exchange
timestamps; no timezone conversion is performed in this diagnostic.

Most days have about 15 candles:

| Candles per date | Value |
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

The one very large gap is 2022-02-25 23:00 to 2022-03-24 09:00, consistent with the
known Russian market halt period. There are only 3 intraday gaps greater than 1h inside
an observed trading day, each missing one hourly bar inside the observed day span:

- 2022-02-24: 07:00 to 09:00
- 2024-02-13: 13:00 to 15:00
- 2025-09-13: 09:00 to 11:00

These are now visible in the report. They are not introduced by cleaning or splitting.

Raw file coverage:

- matching raw files for SBER 1H: 1;
- loaded file: `data/raw/SBER_1H_20200103T0900_20260503T1800.parquet`;
- overlaps between raw files: 0;
- gaps between raw files: 0.

So this run is not silently ignoring earlier raw files.

## Pipeline Loss Accounting

Status: confirmed by code and command run.

For `window=32`, `horizon=1`:

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

Mathematical loss per split:

```text
expected samples = split_len - window_size - horizon + 1
```

For each split, this means 31 candles are not sample targets because of context warmup,
and 1 candle is not a sample target because its future label would leave the split.

All accounting checks are true:

- cleaned begin sorted;
- no duplicate begin;
- no invalid OHLC;
- no missing OHLCV;
- split ranges do not overlap;
- shape rows equal cleaned rows;
- word rows equal cleaned rows;
- valid labels equal `len(df) - horizon`;
- invalid labels are tail-only;
- sentence windows are aligned.

## Next-Word Forecasting

Status: confirmed by code.

New module:

- `ml/src/nlp/word_forecast.py`

New research CLI:

- `ml/scripts/sber_next_word_research.py`

Task:

```text
input:  words[t-L+1], ..., words[t]
target: words[t+1], ..., words[t+K]
```

The clusterer is fit only on train candle shapes. Validation/test words are assigned
through the fitted clusterer. Target future words stay inside their own split.

Implemented baselines:

- persistence: repeat the latest observed word;
- unigram: predict the most frequent train future word;
- Markov-1: direct transition from last context word to each future step;
- TF-IDF logistic regression: per-horizon classifiers over context word n-grams.

Similarity-aware evaluation:

- exact per-horizon accuracy;
- macro-F1 per horizon;
- sequence exact match;
- top-3 accuracy when probabilities exist;
- mean centroid distance;
- soft similarity `exp(-distance / tau)`, where `tau` is the median nonzero train-centroid distance;
- within-nearest-3 accuracy.

Centroids and similarity are derived only from train-fitted clusters.

Soft similarity sanity diagnostics are written to JSON:

- min/median/mean/max nonzero centroid distance;
- random-uniform baseline soft similarity;
- per-result mean predicted-vs-true centroid distance;
- unigram baseline metrics as an ordinary model row.

## Next-Word Results

Status: confirmed by command run.

Command:

```powershell
python ml\scripts\sber_next_word_research.py --output-json data/reports/sber_h1_next_word_research_20260515.json --output-csv data/reports/sber_h1_next_word_research_20260515.csv
```

Selection is validation-only: validation mean accuracy, then validation mean soft similarity,
then deterministic config order.

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

Soft similarity sanity:

| Item | Value |
| --- | ---: |
| Min nonzero centroid distance | 0.2411 |
| Median nonzero centroid distance (`tau`) | 7.5434 |
| Mean nonzero centroid distance | 25.8802 |
| Max centroid distance | 121.8638 |
| Random-uniform val soft similarity | 0.6266 |
| Random-uniform test soft similarity | 0.6255 |
| Unigram val soft similarity, context 32 K=1 | 0.9210 |

So the `0.92-0.95` model soft similarity is not merely an artifact of the exponential
score always being high: random word guesses are much lower. However, unigram is already
high on soft similarity, which means exact accuracy and sequence accuracy remain
important and soft similarity should be interpreted as a shape-proximity diagnostic,
not as a substitute for prediction accuracy.

For `markov1`, `context=32`, `K=3`:

| Horizon | Val exact | Val top-3 | Val within-nearest-3 |
| ---: | ---: | ---: | ---: |
| 1 | 0.2657 | 0.5757 | 0.6132 |
| 2 | 0.2305 | 0.5197 | 0.5954 |
| 3 | 0.2242 | 0.5210 | 0.6178 |

Sequence exact match for K=3 is low: validation 0.0580, test 0.1473.

## Interpretation

Status: interpretation and remaining concerns.

Next-word prediction is better than persistence and unigram on validation exact@1,
but the signal is still weak. The linear TF-IDF classifier did not beat Markov-1 in
this first grid. Multi-step sequence prediction is hard: exact full-sequence matching
falls quickly.

This supports using next-word prediction as a pretext/diagnostic task before making it
a trading decision layer. A reasonable next step is:

1. keep action classification as the downstream target;
2. add next-word prediction as a representation-learning or validation task;
3. only later map predicted word distributions to risk/action, using validation-only
   calibration and a separate trading metric.

Remaining concerns:

- the calendar audit identifies 3 intraday gaps greater than 1h; these are not created
  by the pipeline, but should stay visible in future data-quality reports;
- next-word validation exact@1 is modest, and K=3 sequence exact match is low;
- soft similarity is useful only together with exact/top-k metrics because dominant
  or nearby clusters can score softly well.
