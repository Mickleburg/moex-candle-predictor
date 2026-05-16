# SBER H1: LM-признаки для downstream action classification

Дата: 2026-05-15  
Статус: research-only, без изменения production `/predict` и без final test evaluation.

## 1. Цель

Цель этапа - проверить, может ли candle-word language model быть не только диагностикой продолжения последовательности слов, но и источником признаков для downstream задачи `SELL/HOLD/BUY`.

Проверяемая схема:

```text
raw candles
-> candle shapes
-> train-fitted vocabulary
-> context words
-> train-fitted word LM
-> LM prior/confidence features
-> action classifier
-> SELL/HOLD/BUY
```

Важно: фактические будущие candle words не используются как признаки. Будущее используется только для supervised action label.

## 2. Почему проверяем LM features после vocabulary selection

Vocabulary selection показал, что словарь нельзя выбирать только по perplexity. Маленькие словари могут быть слишком грубыми и легко предсказуемыми за счет dominant word effect. Поэтому downstream experiment должен ответить на другой вопрос: сохраняет ли candle-word LM полезную информацию для действия `SELL/HOLD/BUY`, а не только хорошо угадывает частое следующее слово.

## 3. Использованные vocabularies

Запущены три контрольных словаря:

| vocabulary | роль |
|---|---|
| `shape/kmeans/20` | primary vocabulary из vocabulary selection |
| `shape/kmeans/16` | secondary rolling baseline |
| `shape/gmm_diag/16` | stability/downstream control |

Все словари fit-ятся заново внутри каждого fold только на train range.

## 4. Feature sets

| feature set | состав |
|---|---|
| `base` | sentence features через `cooccurrence_svd` без LM-признаков |
| `lm_scalar` | только scalar LM-признаки: confidence, entropy, margin, self-transition, centroid-distance, beam aggregates |
| `base_lm_scalar` | `base` + scalar LM-признаки |
| `base_lm_proba` | `base` + scalar LM-признаки + полный next-word probability vector |
| `lm_only` | scalar LM-признаки + полный next-word probability vector, без sentence-vectorizer |

Классификаторы: `RidgeClassifier` и `LogisticRegression`.

## 5. Walk-forward folds

Проверены два режима:

| режим | параметры |
|---|---|
| expanding | `max_folds=3` |
| rolling | `train_size=12000`, `val_size=3000`, `step_size=3000`, `max_folds=4` |

В обоих режимах train строго раньше validation, fold ranges не пересекаются, target horizon остается внутри validation fold.

## 6. Leakage guarantees

Подтверждено кодом:

- clusterer fit only на train fold;
- validation words назначаются через train-fitted clusterer;
- `NGramBackoffLanguageModel` fit only на train fold words;
- vectorizer fit only на train action samples;
- action classifier fit only на train samples;
- LM-признаки строятся по context words `w[t-L+1..t]`;
- builder LM-признаков не принимает `Y_future_words` или future target words;
- action labels используются только как target/evaluation;
- selection идет только по validation aggregates;
- test split не используется.

## 7. Action metrics

Для каждого fold/config считаются:

- accuracy;
- balanced accuracy;
- macro-F1;
- precision/recall/F1 для `SELL`, `HOLD`, `BUY`;
- prediction distribution;
- true distribution.

Основная selection metric: validation mean macro-F1. Tie-breakers: worst-fold macro-F1, затем сумма BUY/SELL F1. Test не участвует.

## 8. Calibration и abstention metrics

Для confidence analysis считаются:

- action top-1 confidence;
- action margin top1-top2;
- entropy action probability distribution;
- LM top1 probability;
- LM top3 mass;
- coverage/quality curves.

Пороговые условия:

```text
action_confidence >= 0.40
action_confidence >= 0.50
action_confidence >= 0.60
lm_top1_prob >= 0.50
lm_top3_mass >= 0.80
action_confidence >= 0.50 and lm_top1_prob >= 0.50
```

Эти пороги анализируются только на validation folds. Они не являются production thresholds.

## 9. Результаты base vs LM features

### Expanding

Лучший validation-only config:

| vocabulary | feature set | classifier | macro-F1 mean | std | worst fold | accuracy | BUY F1 | SELL F1 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `shape/gmm_diag/16` | `lm_only` | `ridge` | 0.3961 | 0.0163 | 0.3740 | 0.4595 | 0.2161 | 0.3616 |

Сравнение с base:

| vocabulary | feature set | classifier | macro-F1 mean | worst fold | BUY F1 | SELL F1 |
|---|---|---|---:|---:|---:|---:|
| `shape/gmm_diag/16` | `base` | `logreg` | 0.3907 | 0.3504 | 0.2487 | 0.3131 |
| `shape/gmm_diag/16` | `base` | `ridge` | 0.3894 | 0.3495 | 0.2446 | 0.3128 |
| `shape/kmeans/20` | `base` | `logreg` | 0.3574 | 0.3091 | 0.2354 | 0.2471 |
| `shape/kmeans/16` | `base` | `logreg` | 0.3562 | 0.3002 | 0.2268 | 0.2494 |

Лучшие additive варианты на `shape/gmm_diag/16`:

| feature set | classifier | macro-F1 mean | BUY F1 | SELL F1 |
|---|---|---:|---:|---:|
| `base_lm_proba` | `logreg` | 0.3920 | 0.2494 | 0.3175 |
| `base_lm_proba` | `ridge` | 0.3917 | 0.2525 | 0.3141 |
| `base_lm_scalar` | `logreg` | 0.3915 | 0.2530 | 0.3120 |

Интерпретация: LM-сигнал полезен, но на expanding он сильнее работает как самостоятельный компактный сигнал (`lm_only`), чем как простое добавление к base sentence features.

### Rolling

Лучший validation-only config:

| vocabulary | feature set | classifier | macro-F1 mean | std | worst fold | accuracy | BUY F1 | SELL F1 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `shape/gmm_diag/16` | `lm_only` | `logreg` | 0.4026 | 0.0285 | 0.3693 | 0.4717 | 0.2403 | 0.3435 |

Сравнение с base:

| vocabulary | feature set | classifier | macro-F1 mean | worst fold | BUY F1 | SELL F1 |
|---|---|---|---:|---:|---:|---:|
| `shape/gmm_diag/16` | `base` | `ridge` | 0.3914 | 0.3495 | 0.2265 | 0.3194 |
| `shape/gmm_diag/16` | `base` | `logreg` | 0.3902 | 0.3504 | 0.2222 | 0.3205 |
| `shape/kmeans/20` | `base` | `logreg` | 0.3721 | 0.3091 | 0.2451 | 0.2541 |
| `shape/kmeans/16` | `base` | `ridge` | 0.3662 | 0.3013 | 0.2249 | 0.2568 |

Лучшие additive варианты на `shape/gmm_diag/16`:

| feature set | classifier | macro-F1 mean | BUY F1 | SELL F1 |
|---|---|---:|---:|---:|
| `base_lm_proba` | `ridge` | 0.3974 | 0.2538 | 0.3099 |
| `base_lm_scalar` | `logreg` | 0.3951 | 0.2442 | 0.3125 |
| `base_lm_scalar` | `ridge` | 0.3949 | 0.2428 | 0.3122 |

Интерпретация: rolling подтверждает, что `shape/gmm_diag/16 + lm_only` дает лучший validation macro-F1. Это не доказывает торговую пригодность, но показывает, что LM prior содержит downstream signal.

## 10. Confidence subsets

Для лучших config:

| режим | best config | coverage `action_conf>=0.50` | macro-F1 on covered | coverage `lm_top1>=0.50` | macro-F1 on covered |
|---|---|---:|---:|---:|---:|
| expanding | `shape/gmm_diag/16 lm_only ridge` | 0.196 | 0.3060 | 0.297 | 0.2754 |
| rolling | `shape/gmm_diag/16 lm_only logreg` | 0.327 | 0.3141 | 0.282 | 0.2868 |

Вывод по confidence: текущие confidence-пороги не выделяют subset с лучшим macro-F1. Напротив, качество на covered subset ниже общего. Это похоже на miscalibration и/или усиление dominant/HOLD-like поведения. Confidence полезен как диагностический сигнал, но пока не как готовая abstention logic.

## 11. BUY/SELL analysis

Rolling best `lm_only/logreg` на `shape/gmm_diag/16` улучшает оба action-side класса относительно base `logreg` на том же словаре:

| config | BUY F1 | SELL F1 |
|---|---:|---:|
| `base logreg` | 0.2222 | 0.3205 |
| `lm_only logreg` | 0.2403 | 0.3435 |

Expanding best `lm_only/ridge` улучшает SELL F1, но снижает BUY F1 относительно base:

| config | BUY F1 | SELL F1 |
|---|---:|---:|
| `base ridge` | 0.2446 | 0.3128 |
| `lm_only ridge` | 0.2161 | 0.3616 |

Вывод: LM-признаки не просто увеличивают accuracy через HOLD, но эффект по BUY/SELL нестабилен. Нужно отдельно контролировать BUY/SELL precision/recall перед любыми downstream решениями.

## 12. Ошибки и ограничения

- Это validation-only walk-forward research. Test не использовался.
- `shape/kmeans/20`, выбранный как primary vocabulary для language-model этапа, не оказался лучшим downstream action vocabulary.
- `shape/gmm_diag/16` выглядит сильнее для downstream, но его нужно дополнительно проверить на более широкой сетке и с повтором random_state.
- Confidence thresholds пока не дают качественного covered subset.
- Простое concatenation `base + LM` не дает устойчивого выигрыша над `lm_only`.
- Нет trading backtest и нет production threshold.
- Нет нейросетевых моделей sequence continuation. Это осознанное ограничение этапа.

## 13. Подтверждено запуском

Запущены команды:

```powershell
python ml\scripts\sber_action_lm_features_walk_forward.py `
  --vocab-configs shape:kmeans:20,shape:kmeans:16,shape:gmm:16 `
  --context-size 16 `
  --forecast-horizon 3 `
  --lm-order 2 `
  --lm-alpha 0.1 `
  --fold-mode expanding `
  --max-folds 3 `
  --feature-sets base,lm_scalar,base_lm_scalar,base_lm_proba,lm_only `
  --classifiers ridge,logreg `
  --output-json data/reports/sber_h1_action_lm_features_expanding_20260515.json `
  --output-csv data/reports/sber_h1_action_lm_features_expanding_20260515.csv
```

```powershell
python ml\scripts\sber_action_lm_features_walk_forward.py `
  --vocab-configs shape:kmeans:20,shape:kmeans:16,shape:gmm:16 `
  --context-size 16 `
  --forecast-horizon 3 `
  --lm-order 2 `
  --lm-alpha 0.1 `
  --fold-mode rolling `
  --train-size 12000 `
  --val-size 3000 `
  --step-size 3000 `
  --max-folds 4 `
  --feature-sets base,lm_scalar,base_lm_scalar,base_lm_proba,lm_only `
  --classifiers ridge,logreg `
  --output-json data/reports/sber_h1_action_lm_features_rolling_20260515.json `
  --output-csv data/reports/sber_h1_action_lm_features_rolling_20260515.csv
```

## 14. Вывод

LM-derived features помогают downstream action classification, но текущий результат не выглядит как простое “добавили LM к base и стало лучше”. Более честная формулировка:

- `shape/gmm_diag/16` дает лучший downstream signal среди проверенных словарей;
- `lm_only` превосходит base по validation macro-F1 в expanding и rolling;
- additive `base + LM` улучшает часть baseline вариантов, но не превосходит лучший `lm_only`;
- confidence/abstention пока не готова, потому что high-confidence subsets не дают роста macro-F1;
- `shape/kmeans/20` остается хорошим primary vocabulary для интерпретируемой LM, но для downstream action следующим кандидатом стоит считать `shape/gmm_diag/16`.

Следующий разумный шаг - не переходить сразу к GRU/TCN, а сначала проверить устойчивость `shape/gmm_diag/16 + lm_only` и `shape/gmm_diag/16 + base_lm_proba` на повторных random_state, более аккуратной calibration и per-regime анализе BUY/SELL ошибок.
