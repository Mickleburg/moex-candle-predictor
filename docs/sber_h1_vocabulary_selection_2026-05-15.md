# SBER H1: выбор candle vocabulary, 2026-05-15

## 1. Цель

Цель этапа - выбрать не самый простой и не самый "красивый" словарь, а честный candle vocabulary для дальнейшей language-model и downstream action работы.

Словарь должен быть:

- достаточно предсказуемым для n-gram/backoff LM;
- не слишком грубым;
- не слишком дисбалансным;
- устойчивым на expanding и rolling folds;
- пригодным для интерпретируемого анализа ошибок и уверенности.

Transformer, GRU/LSTM, production artifacts, `/predict` и trading layer в этом проходе не менялись.

## 2. Почему нельзя выбирать словарь только по perplexity

Минимальная perplexity может означать не хороший язык, а слишком грубый словарь. Например, `ohlc/kmeans/8` в expanding sweep имеет очень низкую validation perplexity около `1.56`, но dominant word share около `0.79`, top-3 share около `0.99` и normalized entropy около `0.33`.

Такой словарь легко предсказывать, потому что почти все сводится к нескольким словам. Это не значит, что candle-language representation стала информативной.

Хороший словарь должен балансировать:

- предсказуемость: NLL/perplexity/top-k;
- информативность: entropy/effective vocab size/dominant share;
- устойчивость: fold-to-fold distribution stability;
- практичность: отсутствие большого хвоста редких слов.

## 3. Grid словарей

Expanding run:

```powershell
python ml\scripts\sber_word_lm_walk_forward.py --context-sizes 16 --forecast-horizons 3 --orders 1,2 --shape-variants ohlc,shape --clusterers kmeans,gmm --vocab-sizes 8,12,16,20,24,32 --fold-mode expanding --max-folds 3 --beam-width 3 --min-norm-entropy 0.50 --max-dominant-share 0.55 --max-top3-share 0.80 --output-json data/reports/sber_h1_vocab_selection_expanding_20260515.json --output-csv data/reports/sber_h1_vocab_selection_expanding_20260515.csv
```

Rolling run:

```powershell
python ml\scripts\sber_word_lm_walk_forward.py --context-sizes 16,32 --forecast-horizons 3 --orders 1,2 --shape-variants shape --clusterers kmeans,gmm --vocab-sizes 16,20,24,32 --fold-mode rolling --train-size 12000 --val-size 3000 --step-size 3000 --max-folds 4 --beam-width 3 --min-norm-entropy 0.50 --max-dominant-share 0.55 --max-top3-share 0.80 --output-json data/reports/sber_h1_vocab_selection_rolling_20260515.json --output-csv data/reports/sber_h1_vocab_selection_rolling_20260515.csv
```

По умолчанию для GMM использовался `covariance_type=diag`, `reg_covar=1e-6`, фиксированный `random_state`. Для KMeans использовался `n_init=10` и фиксированный `random_state`.

## 4. Leakage Rules

Подтверждено кодом:

- clusterer fit только на train fold shapes;
- validation words назначаются только через train-fitted clusterer;
- n-gram counts fit только на train fold words;
- vocabulary distribution, transition entropy, self-transition rate и fold statistics считаются по train fold;
- validation используется только для оценки LM;
- actual future validation words не используются как features;
- target words используются только как target/evaluation;
- selection использует только validation aggregates;
- test split не используется.

## 5. Expanding vs Rolling Folds

Expanding:

| Fold | Train | Validation |
| ---: | --- | --- |
| 1 | `[0:12000)` | `[12000:15000)` |
| 2 | `[0:15000)` | `[15000:18000)` |
| 3 | `[0:18000)` | `[18000:21000)` |

Rolling:

| Fold | Train | Validation |
| ---: | --- | --- |
| 1 | `[0:12000)` | `[12000:15000)` |
| 2 | `[3000:15000)` | `[15000:18000)` |
| 3 | `[6000:18000)` | `[18000:21000)` |
| 4 | `[9000:21000)` | `[21000:24000)` |

Rolling проверяет, помогает ли не тащить слишком старую историю в train.

## 6. Constraints для отбора словаря

Vocabulary допустим, если:

```text
normalized_entropy >= 0.50
dominant_word_share <= 0.55
top3_word_share <= 0.80
vocab_size_observed >= 0.80 * vocab_size_requested
```

Если словарь не проходит, JSON содержит `rejection_reason`. Примеры rejected expanding configs:

| Vocabulary | Reason |
| --- | --- |
| `ohlc/kmeans/8` | `normalized_entropy<0.5; dominant_share>0.55; top3_share>0.8` |
| `shape/kmeans/8` | `normalized_entropy<0.5; dominant_share>0.55; top3_share>0.8` |
| `shape/kmeans/12` | `normalized_entropy<0.5; dominant_share>0.55; top3_share>0.8` |
| `ohlc/kmeans/12` | `top3_share>0.8` |
| `shape/kmeans/16` | `top3_share>0.8` |

## 7. Vocabulary Audit Metrics

Для каждого vocabulary config считаются:

- requested/observed vocabulary size;
- empty clusters;
- word counts;
- normalized entropy;
- effective vocabulary size `exp(entropy)`;
- dominant word share;
- top-3/top-5 word share;
- rare word share для слов с частотой `< 1%`;
- transition entropy;
- self-transition rate;
- average outgoing transition entropy;
- fold-to-fold L1/JS stability of word distribution.

## 8. LM Metrics

Token-level:

- accuracy@1;
- top-3/top-5 accuracy;
- MRR;
- token NLL;
- perplexity.

Sequence-level:

- greedy sequence exact match;
- beam contains true sequence;
- sequence NLL;
- centroid trajectory distance.

CSV теперь является candidate vocabulary table, а JSON содержит fold-level LM rows, vocabulary audit rows, confidence analysis и error analysis.

## 9. Confidence Analysis

Для лучшего constrained config confidence считается по first next-token distribution:

- top-1 probability;
- top-3 probability mass;
- entropy distribution;
- margin top1-top2;
- confidence buckets;
- abstention curves.

Expanding best `shape/kmeans/20`:

| Threshold | Coverage | Accuracy@1 |
| --- | ---: | ---: |
| top1 >= 0.25 | 0.9296 | 0.6013 |
| top1 >= 0.35 | 0.7506 | 0.6652 |
| top1 >= 0.50 | 0.5815 | 0.7248 |

Среднее по всем validation samples:

```text
accuracy@1: 0.5753
top3 accuracy: 0.8397
mean top1 probability: 0.5355
mean top3 mass: 0.7980
```

Rolling best `shape/kmeans/16`:

| Threshold | Coverage | Accuracy@1 |
| --- | ---: | ---: |
| top1 >= 0.25 | 0.9629 | 0.6586 |
| top1 >= 0.35 | 0.8703 | 0.6991 |
| top1 >= 0.50 | 0.7200 | 0.7569 |

Интерпретация: confidence полезна. Есть subset с заметно более высокой accuracy, но coverage при строгом threshold падает. Это не выглядит полностью ложной уверенностью, но требуется calibration check перед downstream использованием.

## 10. Error Analysis

Expanding best `shape/kmeans/20`, fold 1:

- самый частый true/predicted pair: `17 -> 17`, count `2144`;
- частые ошибки: `6 -> 17`, `16 -> 17`, `12 -> 17`, `0 -> 17`;
- низкая accuracy у частых слов:
  - word `12`: accuracy `0.0000`, mean NLL `2.6087`;
  - word `0`: accuracy `0.0000`, mean NLL `2.7060`;
  - word `16`: accuracy `0.0181`, mean NLL `2.2711`;
  - word `6`: accuracy `0.0827`, mean NLL `2.0584`.

Rolling best `shape/kmeans/16`, fold 1:

- самый частый true/predicted pair: `8 -> 8`, count `2137`;
- частые ошибки: `3 -> 8`, `8 -> 3`, `0 -> 8`;
- низкая accuracy у частых слов:
  - word `0`: accuracy `0.0127`, mean NLL `2.3585`;
  - word `13`: accuracy `0.0357`, mean NLL `2.5002`;
  - word `3`: accuracy `0.2619`, mean NLL `1.6471`.

Вывод: даже constrained словари остаются дисбалансными внутри отдельных validation fold. Модель часто перекидывает редкие/средние слова в доминирующий word. Это главный риск для downstream signal.

## 11. Лучшие словари

### Лучший по perplexity, но тривиальный

```text
ohlc/kmeans/8
expanding token NLL: 0.4386
perplexity: 1.5575
top3: 0.9990
sequence exact: 0.7257
accepted_by_constraints: false
reason: normalized_entropy<0.5; dominant_share>0.55; top3_share>0.8
```

Это слишком грубый язык.

### Лучший после constraints на expanding

```text
shape/kmeans/20
context_size: 16
forecast_horizon: 3
order: 2
token NLL: 1.2893
token NLL std: 0.2081
perplexity: 3.7061
top3: 0.8397
MRR: 0.7238
sequence exact: 0.3228
normalized entropy: 0.5917
dominant share: 0.4300
top3 share: 0.7270
effective vocab size: 5.8921
```

### Лучший после constraints на rolling

```text
shape/kmeans/16
context_size: 16
forecast_horizon: 3
order: 2
token NLL: 1.0795
token NLL std: 0.2687
perplexity: 3.0598
top3: 0.8862
MRR: 0.7757
sequence exact: 0.4010
normalized entropy: 0.5552
dominant share: 0.4959
top3 share: 0.7871
effective vocab size: 4.8698
```

### Лучший по stability

Среди проверенных accepted configs GMM-словари обычно стабильнее по fold distribution L1, но хуже по LM NLL. Например:

```text
expanding shape/gmm_diag/16
fold_distribution_l1: 1.0799
NLL: 1.6631
perplexity: 5.3579
```

Это хороший контрольный словарь для stability, но не лучший основной candidate.

## 12. Вывод

Рекомендация для следующего downstream action experiment:

```text
primary vocabulary: shape/kmeans/20
secondary rolling baseline: shape/kmeans/16
stability control: shape/gmm_diag/16 или shape/gmm_diag/20
```

Почему `shape/kmeans/20`:

- проходит constraints с запасом по dominant/top3 share;
- лучше на expanding validation среди accepted vocabularies;
- информативнее, чем `shape/kmeans/16`;
- существенно предсказуемее, чем `shape/gmm/20`;
- остается интерпретируемым и не требует сложной модели.

Риски:

- внутри отдельных validation folds все еще есть сильный dominant-word effect;
- редкие слова часто схлопываются в доминирующий prediction;
- fold distribution L1 высокий, значит режимы/распределения слов меняются;
- confidence помогает выделить лучший subset, но нужна calibration проверка.

Переходить к GRU/TCN пока рано. Сначала стоит доработать n-gram baseline:

1. проверить `shape/kmeans/20` в downstream action features;
2. добавить calibration/abstention на validation folds;
3. попробовать более аккуратный smoothing/backoff для order-2;
4. сравнить expanding vs rolling на одном выбранном словаре с большим числом folds.
