# SBER H1: walk-forward research, 2026-05-15

## Цель

Этот этап добавляет expanding-window walk-forward validation для двух исследовательских направлений:

- candle-word forecasting: previous words -> next word(s);
- action classification: previous words -> `SELL/HOLD/BUY`.

Цель этапа - проверить устойчивость во времени, а не максимизировать метрику любой ценой. Test split в этих скриптах не используется для selection.

## Подтверждено кодом

- `walk_forward_ranges` строит ordered expanding folds без пересечения train/validation.
- Clusterers fit-ятся внутри fold только на train candle shapes.
- Validation words назначаются через train-fitted clusterer.
- Vectorizer и classifier в action pipeline fit-ятся только на train fold samples.
- Next-word forecasters fit-ятся только на train fold word targets.
- Centroid distance matrices derived from train-fitted clusterers.
- Markov next-word prior features fit-ятся только по train fold transitions.
- Validation samples держат context и target внутри validation fold.
- Selection использует только validation-fold aggregates.

## Подтверждено запуском

Команды:

```powershell
python ml\scripts\sber_next_word_walk_forward.py --output-json data/reports/sber_h1_next_word_walk_forward_20260515.json --output-csv data/reports/sber_h1_next_word_walk_forward_20260515.csv
python ml\scripts\sber_nlp_walk_forward.py --output-json data/reports/sber_h1_nlp_walk_forward_20260515.json --output-csv data/reports/sber_h1_nlp_walk_forward_20260515.csv
```

Default folds на cleaned SBER 1H frame с 24613 свечами:

| Fold | Train range | Validation range | Train rows | Validation rows |
| ---: | --- | --- | ---: | ---: |
| 1 | `[0:12000)` | `[12000:15000)` | 12000 | 3000 |
| 2 | `[0:15000)` | `[15000:18000)` | 15000 | 3000 |
| 3 | `[0:18000)` | `[18000:21000)` | 18000 | 3000 |

Train всегда строго раньше validation.

## Формулы потерь samples

Для action samples:

```text
samples = fold_len - window_size - horizon + 1
```

При `window_size=32`, `horizon=1` validation fold на 3000 rows дает 2968 samples.

Для next-word samples:

```text
samples = fold_len - context_size - forecast_horizon + 1
```

Эти потери математически неизбежны: часть свечей нужна для warmup context, а tail не может иметь target внутри fold.

## Next-Word Walk-Forward Results

Top validation aggregates:

| Model | Context | K | Folds | Val mean exact | Std | Min | Max | Val soft |
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

## Action Walk-Forward Results

Validation aggregates:

| Config | Markov prior features | Folds | Val macro-F1 | Std | Worst fold | Val accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| best_holdout | false | 3 | 0.3980 | 0.0275 | 0.3611 | 0.4666 |
| best_holdout_markov_features | true | 3 | 0.3973 | 0.0262 | 0.3615 | 0.4657 |
| kmeans_tfidf_ridge | false | 3 | 0.3448 | 0.0405 | 0.2884 | 0.4352 |

Best by validation-fold selection:

```text
best_holdout
```

## Почему прежние метрики слабые

- `Markov-1` - baseline по последнему слову, а не полноценная language model. Он оценивает локальную инерцию, но не строит богатую conditional distribution для всей будущей фразы.
- Exact sequence match для `K > 1` быстро падает: даже при приемлемой token accuracy ошибка на любом шаге ломает всю sequence.
- Candle words - это cluster IDs формы свечи, а не натуральные слова с устойчивой семантикой.
- Soft similarity часто высока даже у простых baselines, потому что многие cluster centroids близки по форме. Ее нельзя использовать как главный selection metric.
- Fold variance около 0.06-0.10 для next-word metrics указывает на смену рыночных режимов.
- Текущая модель скорее ловит локальную инерцию candle shapes, чем устойчивую "грамматику свечей".

## Интерпретация

Action config из holdout-этапа остается лучшим среди малой walk-forward проверки: mean macro-F1 0.3980 против holdout validation около 0.4067. Это поддерживает направление исследования, но не доказывает торговую пригодность.

Markov next-word prior features не улучшили action classifier в минимальном эксперименте: 0.3973 против 0.3980 macro-F1. Это не закрывает идею pretext features, но текущая простая версия пользы не показала.

Next-word baseline `markov1` лучше persistence/unigram по mean exact, но variance по folds слишком велика для сильного вывода.

## Ограничения

- Использовано только 3 folds, чтобы runtime оставался умеренным.
- Final test evaluation не выполнялся.
- Next-word path на этом этапе еще не был полноценной sequence language model.
- Trading layer не строился.

## Следующие шаги

1. Перейти от per-horizon next-word baselines к n-gram/backoff language model.
2. Оценивать `P(w[t+1..t+K] | words[t-L+1..t])` через token NLL, perplexity, top-k, MRR и beam sequence metrics.
3. Проверить качество словаря candle words: clusterer, vocabulary size, entropy, transition entropy и validation perplexity.
4. Не переходить к Transformer/LSTM до сильного честного n-gram baseline.
