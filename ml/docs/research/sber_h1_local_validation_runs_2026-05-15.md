# SBER H1: локальные validation-only прогоны narrow-сетки

## 1. Что было добито локально

Пользователь локально завершил полную narrow-сетку, которую Codex ранее сократил из-за runtime:

```text
vocab: shape/gmm:20, shape/gmm:16
features: lm_regime, lm_regime_proba
weights: balanced, action_boost_1.2, action_boost_1.5
modes: argmax, global
random_states: 7,13,21,42,100
```

Это validation-only результаты. Test split не использовался для выбора, сравнения или ретюнинга.

## 2. Лучший честный validation config

Лучшим честным config остался:

```text
shape/gmm_diag/20 + lm_regime + action_boost_1.2 + argmax

macro-F1:    0.4238
worst:       0.4057
BUY F1:      0.3603
SELL F1:     0.3281
action_rate: 0.5983
```

Это подтверждает, что предыдущий research-primary candidate не был вытеснен более узкими вариантами вокруг него.

## 3. Близкие варианты

`shape/gmm_diag/16` близок, но не лучше по mean:

```text
shape/gmm_diag/16 + lm_regime + action_boost_1.2 + argmax

macro-F1:    0.4231
worst:       0.4100
BUY F1:      0.3617
SELL F1:     0.3252
action_rate: 0.5962
```

`shape/gmm_diag/16` можно держать как control vocabulary, но он не заменяет primary `shape/gmm_diag/20`.

## 4. lm_regime_proba

Полный next-word probability vector поверх regime features не дал надежного прироста:

```text
shape/gmm_diag/20 + lm_regime_proba + action_boost_1.2 + argmax

macro-F1: 0.4233
worst:    0.4074
BUY F1:   0.3560
SELL F1:  0.3296
```

Вывод: добавление probability vector не является текущим главным рычагом улучшения.

## 5. Threshold modes

`global thresholds` хуже `argmax` в честной validation-схеме.

`oracle_global` находится около `0.428-0.430`, то есть даже leakage/oracle upper bound показывает небольшой запас от thresholding. Это важный сигнал: дальнейшая возня с threshold grid, temperature и BUY/SELL порогами уже почти исчерпала потенциал.

## 6. Untouched test

Final untouched test уже был выполнен один раз для frozen candidate:

```text
test macro-F1 = 0.4055
BUY F1        = 0.2553
SELL F1       = 0.3113
HOLD F1       = 0.6499
```

Test больше нельзя использовать для выбора новых вариантов, подбора параметров или сравнения новых candidates.

## 7. Вывод

Текущая ветка LM/action threshold tuning почти исчерпана:

- `gmm16/gmm20` не дает качественного скачка;
- `lm_regime_proba` не дает надежного прироста;
- `global thresholds` хуже `argmax`;
- `oracle_global` показывает малый остаточный запас.

Следующий осмысленный шаг - менять рычаг улучшения: target design, continuous past-only features и более сильные baseline-модели без нейросетей. Именно для этого добавляется отдельный validation-only research path.
