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

## Examples

Согласованные examples лежат в `contracts/examples/`.

Они проверяют demo path:

```text
candle_batch + market_snapshot
ml_prediction + llm_analysis -> aggregated_signal
aggregated_signal + portfolio_snapshot -> risk_decision
risk_decision -> order_request -> execution_report
agent_cycle_result references selected/rejected candidates
```

## Проверка

```powershell
python scripts\validate_contracts.py
```

Скрипт проверяет:

- валидный JSON для всех schemas и examples;
- JSON Schema validation, если установлен пакет `jsonschema`;
- совместимость вероятностей `buy/hold/sell`;
- связку ML -> aggregator -> risk -> execution;
- `config/supported_tickers.yaml` для `SBER`, `GAZP`, `LKOH`.

Если `jsonschema` не установлен, базовые JSON и cross-contract checks все равно выполняются, а schema-vs-example validation пропускается с понятным сообщением.
