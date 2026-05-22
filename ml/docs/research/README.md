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

## Новый отчет

- `sber_h1_triple_barrier_seed_robustness_2026-05-15.md` - проверка устойчивости текущего triple-barrier candidate по seeds `7,13,21,42,100`.

Seed robustness подтвердил, что `random_state=42` не был случайным выбросом:

```text
triple_barrier:h3:w12:up1.25:down1.25
continuous_regime
extra_trees:depth=none:leaf=20:maxfeat=sqrt
class_weight=none
mean macro-F1 over seeds=0.4685
worst seed macro-F1=0.4676
worst fold macro-F1=0.4522
```

Этот результат по-прежнему validation-only и не является production artifact.
