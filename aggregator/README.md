# Aggregator Block

`aggregator/` - scaffold будущего блока late fusion.

Назначение:

- принять `ml_prediction`;
- принять `llm_analysis`;
- объединить вероятности;
- вернуть `aggregated_signal`.

Начальная схема весов:

```text
ML:  0.75
LLM: 0.25
```

Aggregator не отправляет orders и не проверяет финальные risk limits. Он должен выбирать не просто максимальную вероятность, а risk-adjusted expected edge. Финальное разрешение делает `risk_manager/`.
