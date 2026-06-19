# SBER H1: закрепление action-кандидата и final test evaluation

## 1. Цель

Цель этапа - остановить расширение research-сетки, проверить устойчивость текущего лучшего кандидата рядом с найденной областью параметров и один раз оценить замороженный кандидат на untouched test split.

В этом этапе не менялся production `/predict`, не собирался production artifact bundle, не запускался trading backtest и не использовались нейросетевые модели.

## 2. Почему переходим к frozen candidate

Предыдущий честный rolling nested результат для research-primary config:

```text
shape/gmm_diag/20 + lm_regime + logreg
class_weight = action_boost_1.2
action_horizon = 1
decision = argmax

rolling nested outer validation:
macro-F1 mean = 0.4265
worst fold    = 0.4223
BUY F1        = 0.3618
SELL F1       = 0.3306
action rate   = 0.5973
```

Старый baseline был `shape/gmm_diag/20 + lm_only + logreg + balanced + global thresholds` с rolling nested macro-F1 `0.4187`.

## 3. Узкая сетка вокруг best config

Полная запрошенная narrow-сетка:

```text
vocab: shape:gmm:20, shape:gmm:16
feature_set: lm_regime, lm_regime_proba
class_weight: balanced, action_boost_1.2, action_boost_1.5
decision: argmax, global
random_states: 7,13,21,42,100
```

Этот прогон уперся в timeout до записи отчета. Поэтому выполнена сокращенная, но содержательная проверка:

```text
vocab: shape:gmm:20
feature_set: lm_regime, lm_regime_proba
class_weight: action_boost_1.2, action_boost_1.5
decision: argmax
random_states: 7,13,21,42,100
folds: rolling, 4 folds
```

Дополнительно отдельно проверен контрольный словарь:

```text
vocab: shape:gmm:16
feature_set: lm_regime
class_weight: action_boost_1.2
decision: argmax
random_states: 7,13,42
folds: rolling, 4 folds
```

## 4. Seed robustness

Для `shape/gmm_diag/20 + lm_regime + action_boost_1.2 + argmax`:

```text
mean macro-F1 across seed x fold = 0.4238
worst row macro-F1              = 0.4057
std across seeds                = 0.0021
std across folds                = 0.0059
worst seed mean                 = 0.4211, seed 7
worst fold mean                 = 0.4150, fold 4
BUY F1 mean                     = 0.3603
SELL F1 mean                    = 0.3281
HOLD F1 mean                    = 0.5829
action rate mean                = 0.5983
```

По seed:

```text
seed 7:   mean 0.4211, worst 0.4057
seed 13:  mean 0.4214, worst 0.4127
seed 21:  mean 0.4250, worst 0.4175
seed 42:  mean 0.4265, worst 0.4223
seed 100: mean 0.4247, worst 0.4169
```

Вывод: seed 42 действительно удачный, но не выглядит одиночным выбросом. Основная нестабильность по-прежнему сильнее связана с fold/regime, чем с initialization.

## 5. Проверка lm_regime_proba

Сравнение при `shape/gmm_diag/20 + action_boost_1.2 + argmax`:

```text
lm_regime:       macro-F1 0.4238, worst 0.4057, BUY F1 0.3603, SELL F1 0.3281, action rate 0.5983
lm_regime_proba: macro-F1 0.4233, worst 0.4074, BUY F1 0.3560, SELL F1 0.3296, action rate 0.5889
```

Полный next-word probability vector не дал надежного прироста. Он немного улучшил worst row, но снизил mean macro-F1 и BUY F1.

## 6. Проверка action_boost_1.5

При `shape/gmm_diag/20 + argmax`:

```text
lm_regime + action_boost_1.5:
macro-F1 0.4072
BUY F1 0.3814
SELL F1 0.3354
HOLD F1 0.5047
action rate 0.7243

lm_regime_proba + action_boost_1.5:
macro-F1 0.4088
BUY F1 0.3779
SELL F1 0.3411
HOLD F1 0.5073
action rate 0.7202
```

`action_boost_1.5` улучшает BUY/SELL, но делает модель слишком агрессивной: action rate около `0.72`, HOLD F1 падает примерно до `0.50`, macro-F1 существенно ниже. Это over-action, а не честное улучшение качества.

## 7. Контроль shape/gmm_diag/16

Для `shape/gmm_diag/16 + lm_regime + action_boost_1.2 + argmax` на seed `7,13,42`:

```text
mean macro-F1 = 0.4231
worst row     = 0.4100
BUY F1        = 0.3590
SELL F1       = 0.3296
action rate   = 0.5969
```

`shape/gmm_diag/16` близок и имеет чуть лучше worst row в сокращенном контроле, но mean ниже, а ранее primary candidate был выбран по более полному контексту. Он остается control config, но не заменяет `shape/gmm_diag/20`.

## 8. Ensemble

Ensemble не реализован в этом проходе.

Причины:

- полная narrow-сетка с seed/vocab/threshold уже превысила runtime и завершилась timeout;
- после выбора frozen candidate был выполнен untouched test, поэтому добавлять новый ensemble-кандидат и затем сравнивать его с test было бы нарушением дисциплины frozen evaluation;
- корректный ensemble должен быть оценен отдельным validation-only этапом до нового test.

## 9. Frozen candidate

Замороженный кандидат выбран до final test:

```text
vocabulary: shape/gmm_diag/20
feature_set: lm_regime
classifier: logreg
class_weight: action_boost_1.2
decision: argmax
action_horizon: 1
lm_context: 16
lm_forecast_horizon: 3
lm_order: 2
lm_alpha: 0.1
random_state policy: fixed:42
threshold policy: none
calibration policy: not used for argmax decision
```

Выбор основан только на rolling nested outer validation. Test при выборе не использовался.

## 10. Final test evaluation

Untouched test evaluation выполнен один раз.

Схема:

```text
chronological split:
train: 0..17229
val:   17229..20921
test:  20921..24613

development = train + val = 0..20921
inner_train = 0..18421
calibration = 18421..20921
test        = 20921..24613
```

Так как frozen decision = `argmax`, calibration не используется для выбора thresholds. Regime thresholds/standardization и все train-derived statistics fit-ятся только на inner_train/development side, не на test.

Test metrics:

```text
macro-F1          = 0.4055
accuracy          = 0.4825
balanced accuracy = 0.4075
BUY F1            = 0.2553
SELL F1           = 0.3113
HOLD F1           = 0.6499
action rate       = 0.4760
n test samples    = 3660
```

Prediction distribution:

```text
SELL: 28.55%
HOLD: 52.40%
BUY:  19.04%
```

True distribution:

```text
SELL: 22.70%
HOLD: 55.46%
BUY:  21.83%
```

Degradation vs rolling validation:

```text
rolling nested macro-F1 = 0.4265
test macro-F1           = 0.4055
delta                   = -0.0210
```

Это заметная деградация. По правилу этапа test не используется для ретюнинга.

## 11. Regime failure analysis

### Volatility

```text
low_vol:  n=2938, macro-F1=0.3877, BUY F1=0.1885, SELL F1=0.2876, HOLD F1=0.6870, action rate=0.3734
mid_vol:  n=517,  macro-F1=0.3327, BUY F1=0.3104, SELL F1=0.3867, HOLD F1=0.3011, action rate=0.8627
high_vol: n=205,  macro-F1=0.2958, BUY F1=0.4478, SELL F1=0.3472, HOLD F1=0.0923, action rate=0.9707
```

В high/mid volatility модель почти полностью уходит в action и теряет HOLD.

### Trend

```text
downtrend: macro-F1=0.3954, BUY F1=0.2445, SELL F1=0.3723, HOLD F1=0.5693, action rate=0.6087
flat:      macro-F1=0.3866, BUY F1=0.1965, SELL F1=0.2685, HOLD F1=0.6948, action rate=0.3628
uptrend:   macro-F1=0.4161, BUY F1=0.3587, SELL F1=0.3255, HOLD F1=0.5640, action rate=0.6398
```

Лучший trend-regime на test - uptrend. Flat остается слабым для BUY/SELL.

### Session

```text
early:  macro-F1=0.2459, BUY F1=0.0297, SELL F1=0.3990, HOLD F1=0.3090, action rate=0.7812
middle: macro-F1=0.3963, BUY F1=0.3647, SELL F1=0.1746, HOLD F1=0.6495, action rate=0.4846
late:   macro-F1=0.3719, BUY F1=0.2564, SELL F1=0.0667, HOLD F1=0.7926, action rate=0.1912
```

Session bias очень сильный: early почти не ловит BUY, late почти не ловит SELL.

### LM uncertainty

```text
low_entropy:  macro-F1=0.3761, BUY F1=0.1708, SELL F1=0.2170, HOLD F1=0.7405, action rate=0.2240
mid_entropy:  macro-F1=0.3086, BUY F1=0.3205, SELL F1=0.3852, HOLD F1=0.2201, action rate=0.8864
high_entropy: macro-F1=0.2390, BUY F1=0.3174, SELL F1=0.3485, HOLD F1=0.0510, action rate=0.9862
```

LM entropy подтверждает прошлый вывод: высокая уверенность/низкая энтропия не является прямым фильтром качества action signal; high entropy превращается в почти сплошной action и ломает HOLD.

## 12. Что подтвердилось

- `lm_regime + action_boost_1.2` остается лучшим среди проверенных primary-вариантов.
- Seed variance небольшая: `std across seeds = 0.0021`.
- `lm_regime_proba` не дает надежного прироста.
- `action_boost_1.5` переусиливает BUY/SELL и разрушает HOLD.
- Основная проблема не initialization, а режимная нестабильность и class/objective mismatch.

## 13. Что не подтвердилось

- Не подтвердилось, что validation-quality `0.4265` переносится на untouched test без заметной деградации.
- Не подтвердилось, что полный next-word probability vector сам по себе улучшает action model.
- Не подтвердилось, что более сильный action boost дает лучший общий classifier.
- Ensemble не проверен в этом этапе.

## 14. Leakage guarantees

- Test не использовался для выбора vocabulary, feature set, class weight, seed или decision rule.
- Clusterer fit только на inner_train/development side.
- Word LM counts fit только на inner_train.
- Regime thresholds и standardization fit только на inner_train.
- Action classifier fit только на inner_train.
- Calibration split не использовался для classifier fit.
- Test использовался только для одного report-only evaluation.
- Actual future candle words не использовались как features.
- Future return использовался только как supervised action target.

## 15. Ограничения

- Test result относится только к одному frozen candidate.
- Frozen candidate показал честную деградацию; ретюнинг по test не выполнялся.
- Ensemble не реализован и должен проверяться только в новом validation-only цикле.
- Нет production artifact bundle.
- Нет торгового backtest и нет claims о торговой пригодности.

## 16. Следующий шаг

Если цель - production research artifact, текущий frozen candidate можно сохранить как baseline-кандидат, но не как финально сильную модель: test macro-F1 `0.4055` ниже rolling validation на `0.0210`, а BUY F1 просел до `0.2553`.

Перед production artifact разумнее сделать еще один research step без test:

- nested validation-only ensemble для seed/vocab averaging;
- session-aware или volatility-aware objective/thresholding, но только через nested protocol;
- отдельная работа с HOLD в high-volatility/high-entropy regimes;
- проверка более устойчивого target design для horizon 1 против horizon 3 без использования test.

Переходить к GRU/TCN пока рано: n-gram/LM-derived pipeline еще не исчерпал validation-only ensemble и objective design.
