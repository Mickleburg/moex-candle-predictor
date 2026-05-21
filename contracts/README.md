# Shared Contracts

`contracts/` содержит JSON Schema для будущих блоков торгового агента.

Эти контракты описывают интерфейсы между блоками:

```text
backend/data -> ml_prediction
backend/data -> llm_analysis
ml_prediction + llm_analysis -> aggregated_signal
aggregated_signal + portfolio_snapshot -> risk_decision
risk_decision -> order_request
order_request -> execution_report
agent -> agent_cycle_result
```

Схемы являются архитектурным scaffold. Они не означают, что соответствующие runtime-блоки уже реализованы.
