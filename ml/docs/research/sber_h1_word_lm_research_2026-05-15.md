# SBER H1: candle-word language model research, 2026-05-15

## 1. Цель

Цель этого прохода - превратить next-word prediction из набора отдельных baselines в честную задачу продолжения последовательности candle words:

```text
input:
w[t-L+1], ..., w[t]

target:
w[t+1], ..., w[t+K]
```

Модель должна оценивать:

```text
P(w[t+1], ..., w[t+K] | w[t-L+1], ..., w[t])
```

Этот этап не строит trading strategy layer и не меняет production `/predict`.

## 2. Почему предыдущие метрики были слабыми

- `Markov-1` из прежнего next-word path был сильным baseline, но не полноценной language model. Он в основном ловит локальную инерцию последнего candle word.
- Exact sequence match быстро падает с ростом `K`: если token accuracy умеренная, вероятность угадать всю последовательность уменьшается почти мультипликативно.
- Candle words - это кластеры форм свечей. Они похожи на "слова" только по роли в pipeline, но не имеют стабильной семантики натурального языка.
- Soft similarity слишком мягкая: похожие candle-shape centroids дают высокий score даже при неверном exact word.
- Walk-forward variance показывает смену рыночных режимов. Один holdout split может выглядеть лучше или хуже только из-за выбранного периода.
- Текущий сигнал больше похож на краткосрочную shape inertia, чем на устойчивую "грамматику свечей".

## 3. Почему Markov-1 - baseline, а не полноценная языковая модель

Markov-1 использует только последний word:

```text
P(w[t+1] | w[t])
```

Для multi-step прогноза этого мало. Нужна модель, которая:

- использует более длинный context suffix;
- дает probability distribution на каждом шаге;
- умеет backoff на короткий context, если длинный n-gram не встречался в train;
- оценивает NLL/perplexity всей target sequence;
- декодирует продолжение без actual future words.

## 4. Постановка sequence continuation

Добавлен модуль:

```text
ml/src/nlp/word_lm.py
```

Реализовано:

- additive-smoothed n-gram LM;
- Markov orders `1..N`;
- backoff на более короткий context;
- greedy decoding;
- beam search decoding для sequence metrics;
- teacher-forced token probability evaluation для NLL/top-k/MRR.

Teacher forcing используется только для оценки вероятности фактической sequence. Это не feature leakage: модель не получает actual future words при fit или free-running decode.

## 5. Leakage Rules

Подтверждено кодом:

- clusterer fit только на train fold shapes;
- validation words assigned только через train-fitted clusterer;
- n-gram counts fit только на train fold words;
- vocabulary distribution и transition entropy считаются только на train fold;
- centroid distances derived from train-fitted clusterer;
- validation target words используются только как target/evaluation;
- generated sequence строится без actual future words;
- selection использует только validation aggregates;
- test split не используется.

## 6. Folds

Default folds:

| Fold | Train range | Validation range | Train rows | Validation rows |
| ---: | --- | --- | ---: | ---: |
| 1 | `[0:12000)` | `[12000:15000)` | 12000 | 3000 |
| 2 | `[0:15000)` | `[15000:18000)` | 15000 | 3000 |
| 3 | `[0:18000)` | `[18000:21000)` | 18000 | 3000 |

Train всегда раньше validation, пересечений нет.

## 7. Vocabulary Analysis

Команда walk-forward дополнительно пишет vocabulary audit в JSON. Проверялись:

- shape variants: `ohlc`, `shape`;
- clusterers: `kmeans`, `gmm`;
- vocabulary sizes: `8,16,20,32,48,64`;
- metrics: word distribution entropy, dominant word share, transition entropy, validation next-token perplexity и top-k accuracy.

Top rows по validation next-token perplexity:

| Shape | Clusterer | Vocab | Norm entropy | Dominant share | Transition entropy | Next-token perplexity | Top-3 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ohlc | kmeans | 8 | 0.3312 | 0.7915 | 0.6398 | 1.5572 | 0.9990 |
| shape | kmeans | 8 | 0.3693 | 0.7486 | 0.6795 | 1.5869 | 0.9959 |
| shape | kmeans | 16 | 0.5210 | 0.5086 | 1.2881 | 2.6951 | 0.9269 |
| shape | gmm | 8 | 0.6726 | 0.3758 | 1.0685 | 2.7232 | 0.9657 |
| ohlc | kmeans | 16 | 0.5634 | 0.4347 | 1.4442 | 3.3516 | 0.9032 |
| shape | kmeans | 20 | 0.5917 | 0.4300 | 1.5962 | 3.7054 | 0.8398 |
| shape | gmm | 20 | 0.8194 | 0.2143 | 2.0331 | 7.0465 | 0.6594 |

Интерпретация:

- Очень маленькие vocabularies дают низкую perplexity, но часто за счет доминирующего слова. Например, `ohlc/kmeans/8` имеет dominant share около 79%.
- Более сбалансированные словари, например `shape/gmm/20`, труднее предсказывать, но они менее тривиальны.
- Нельзя выбирать словарь только по минимальной perplexity: нужно учитывать entropy/dominant share, иначе модель может "хорошо" предсказывать слишком грубый язык.

## 8. Language-Model Metrics

Token-level:

- `accuracy@1`;
- `top-3 accuracy`;
- `top-5 accuracy`;
- macro-F1;
- mean reciprocal rank;
- token NLL;
- perplexity;
- centroid distance;
- soft similarity.

Sequence-level:

- greedy sequence exact match;
- beam contains true sequence;
- sequence NLL;
- mean token NLL;
- average centroid trajectory distance.

Selection делается по validation mean token NLL, затем validation perplexity, затем validation top-k. Test не используется.

## 9. Walk-Forward Results

Команда:

```powershell
python ml\scripts\sber_word_lm_walk_forward.py --context-sizes 16,32 --forecast-horizons 3,5 --orders 1,2,3 --output-json data/reports/sber_h1_word_lm_walk_forward_20260515.json --output-csv data/reports/sber_h1_word_lm_walk_forward_20260515.csv
```

Top validation aggregates для основного словаря `shape/gmm/20`:

| Context | K | Order | Alpha | Token NLL | Perplexity | Top-3 | Sequence exact |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 3 | 1 | 0.1 | 1.9170 | 6.9216 | 0.6595 | 0.0949 |
| 16 | 5 | 1 | 0.1 | 1.9172 | 6.9229 | 0.6595 | 0.0276 |
| 32 | 5 | 1 | 0.1 | 1.9175 | 6.9241 | 0.6594 | 0.0277 |
| 32 | 3 | 1 | 0.1 | 1.9177 | 6.9253 | 0.6594 | 0.0946 |
| 16 | 3 | 2 | 0.1 | 1.9323 | 7.0510 | 0.6592 | 0.1045 |
| 16 | 5 | 2 | 0.1 | 1.9326 | 7.0530 | 0.6592 | 0.0329 |
| 16 | 3 | 3 | 0.1 | 2.1406 | 8.7866 | 0.6175 | 0.1004 |

## 10. Лучший validation-only config

```text
shape_variant: shape
clusterer: gmm
vocabulary_size: 20
context_size: 16
forecast_horizon: 3
markov_order: 1
smoothing_alpha: 0.1
```

Validation aggregate:

```text
token NLL mean: 1.9170
token NLL std: 0.1925
perplexity mean: 6.9216
perplexity std: 1.2346
top-3 mean: 0.6595
sequence exact mean: 0.0949
soft similarity mean: 0.9291
```

## 11. Что получилось / что не получилось

Получилось:

- задача оформлена как sequence continuation;
- multi-step `K=3` и `K=5` оцениваются явно;
- LM считает probability/NLL/perplexity всей target sequence;
- добавлены beam metrics;
- vocabulary audit показывает trade-off между простотой словаря и предсказуемостью.

Не получилось:

- более высокие orders `2` и `3` не улучшили validation NLL на текущем словаре;
- sequence exact остается низким, особенно при `K=5`;
- top-3 около 0.66 говорит, что модель сужает пространство вариантов, но не дает сильного точного прогноза;
- soft similarity остается вспомогательной метрикой, не selection target.

## 12. Следующие шаги

1. Подобрать словарь не только по perplexity, но и по entropy/dominant share.
2. Проверить `shape/kmeans/16-20` и `shape/gmm/16-32` как более честный trade-off.
3. Добавить rolling-window вариант folds, чтобы оценить деградацию при старом train history.
4. Попробовать order-2/3 с stronger smoothing/backoff variants, но не переходить к Transformer/LSTM до сильного n-gram baseline.
5. Если LM будет использоваться как pretext, передавать downstream только predicted distributions/features, а не actual future words.
