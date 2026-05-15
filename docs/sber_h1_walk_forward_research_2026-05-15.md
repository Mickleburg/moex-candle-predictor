# SBER H1 Walk-Forward Research, 2026-05-15

## Scope

This pass adds expanding-window walk-forward validation for two research paths:

- candle-word forecasting: previous words -> next word(s);
- action classification: previous words -> `SELL/HOLD/BUY`.

The goal is stability checking, not maximum score tuning. No test split is used
for selection in these scripts.

## Fold Design

Default folds use the cleaned SBER 1H frame with 24,613 candles:

| fold | train range | validation range | train rows | validation rows |
| ---: | --- | --- | ---: | ---: |
| 1 | `[0:12000)` | `[12000:15000)` | 12000 | 3000 |
| 2 | `[0:15000)` | `[15000:18000)` | 15000 | 3000 |
| 3 | `[0:18000)` | `[18000:21000)` | 18000 | 3000 |

Train is always strictly before validation. The default is expanding-window
validation, implemented by `walk_forward_ranges` in `ml/src/data/split.py`.

## Leakage Rules

Confirmed by code:

- clusterers are fit per fold on train candle shapes only;
- validation words are assigned by train-fitted clusterers;
- sentence/vectorizer models are fit on train fold samples only;
- action classifiers are fit on train fold labels only;
- next-word forecasters are fit on train fold word targets only;
- centroid distance matrices are derived from train-fitted clusterers;
- Markov next-word prior features are fit from train fold transitions only;
- validation samples keep their context and target inside the validation fold;
- selection uses validation-fold aggregates only.

For action samples, the expected count per fold is:

```text
samples = fold_len - window_size - horizon + 1
```

With `window_size=32` and `horizon=1`, a 3000-row validation fold yields
2968 samples.

For next-word samples, the expected count per fold is:

```text
samples = fold_len - context_size - forecast_horizon + 1
```

## Next-Word Walk-Forward Results

Command:

```powershell
python ml\scripts\sber_next_word_walk_forward.py --output-json data/reports/sber_h1_next_word_walk_forward_20260515.json --output-csv data/reports/sber_h1_next_word_walk_forward_20260515.csv
```

Top validation aggregates:

| model | context | K | folds | val mean exact | std | min | max | val soft |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| markov1 | 16 | 1 | 3 | 0.3657 | 0.0707 | 0.2802 | 0.4534 | 0.9292 |
| markov1 | 32 | 1 | 3 | 0.3650 | 0.0703 | 0.2800 | 0.4522 | 0.9291 |
| markov1 | 16 | 3 | 3 | 0.3373 | 0.0599 | 0.2638 | 0.4105 | 0.9294 |
| markov1 | 32 | 3 | 3 | 0.3338 | 0.0600 | 0.2630 | 0.4098 | 0.9291 |
| tfidf_logreg | 16 | 1 | 3 | 0.3061 | 0.1062 | 0.2245 | 0.4561 | 0.9151 |
| persistence | 16 | 1 | 3 | 0.2939 | 0.0624 | 0.2242 | 0.3757 | 0.9161 |

Best by validation-fold selection:

```text
markov1, context_size=16, forecast_horizon=1
```

Interpretation:

- Markov-1 beats persistence and unigram on mean exact accuracy across folds.
- The signal is not regime-stable: std is about 0.071 for the best K=1 result.
- Soft similarity remains high for multiple baselines, so exact/top-k metrics are
  still the primary diagnostic.

## Action Walk-Forward Results

Command:

```powershell
python ml\scripts\sber_nlp_walk_forward.py --output-json data/reports/sber_h1_nlp_walk_forward_20260515.json --output-csv data/reports/sber_h1_nlp_walk_forward_20260515.csv
```

Validation aggregates:

| config | Markov prior features | folds | val macro-F1 | std | worst fold | val accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| best_holdout | false | 3 | 0.3980 | 0.0275 | 0.3611 | 0.4666 |
| best_holdout_markov_features | true | 3 | 0.3973 | 0.0262 | 0.3615 | 0.4657 |
| kmeans_tfidf_ridge | false | 3 | 0.3448 | 0.0405 | 0.2884 | 0.4352 |

Best by validation-fold selection:

```text
best_holdout
```

Interpretation:

- The previous holdout-selected NLP configuration remains the strongest among
  the small walk-forward set.
- Markov next-word prior features do not improve this configuration in the
  current minimal experiment.
- The weaker KMeans/TF-IDF reference confirms that the better holdout result was
  not only a single split artifact, but stability is still modest.

## Holdout vs Walk-Forward

The earlier holdout validation macro-F1 for the best action config was around
0.4067. Walk-forward mean macro-F1 is 0.3980 across three validation periods.
These numbers are close enough to keep the direction alive, but not strong
enough to claim a production trading edge.

The earlier next-word holdout best used one validation segment. Walk-forward
shows Markov-1 remains the most credible simple baseline, but fold variance is
large. Future work should report per-period degradation before trying larger
models.

## Remaining Concerns

- Only three folds are used by default to keep runtime modest.
- No final holdout/test evaluation is performed in these walk-forward scripts.
- Next-word soft similarity is useful as a shape-distance diagnostic, but it is
  too forgiving to be the selection metric by itself.
- The Markov prior feature experiment is intentionally minimal: it uses only
  train-fitted transition probabilities from the current word, plus entropy and
  expected centroid distance.

## Next Steps

1. Add a fourth or fifth fold or a rolling train window to test late-regime
   degradation.
2. Keep Markov-1 as the next-word baseline before adding richer sequence models.
3. Try calibration/thresholding for the action classifier using validation folds
   only.
4. If next-word features are revisited, test them as a pretext representation,
   not as actual future-word leakage.
