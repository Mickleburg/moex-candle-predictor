# ML Research Reports

Здесь лежат research-отчеты, перенесенные из корневого `docs/`.

Правила чтения результатов:

- для выбора модели использовать только aggregate metrics по folds;
- fold-level rows использовать только для диагностики;
- test split не использовать для подбора;
- validation-only результаты не считать production-ready;
- trading claims не делать без отдельного backtest/paper trading.

Текущий лучший validation-only candidate:

```text
triple_barrier:h3:w12:up1.25:down1.25
continuous_regime
extra_trees:depth=none:leaf=20:maxfeat=sqrt
class_weight=none
mean macro-F1=0.4695
```
