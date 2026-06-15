# Shared Contracts

`contracts/` содержит JSON Schema для блоков торгового агента.

> **V2 (2026-06-15):** текущие схемы ещё описывают СТАРУЮ late-fusion-постановку. На этапе
> заморозки контрактов V2 переопределяем: `llm_analysis` → новостные ПРИЗНАКИ (не buy/hold/sell);
> `aggregated_signal` → кросс-секционный ранг/скор по тикерам; новый `feature_bundle`
> (`[quant ⊕ news]`). Блок `aggregator` удалён (ранняя фьюжн). См. `docs/ARCHITECTURE_V2.md`.

Интерфейсы между блоками (V2 — целевые):

```text
backend/data -> candle_batch + market context
news -> llm_analysis (новостные признаки)
candle_batch ⊕ llm_analysis -> ml: кросс-секционный aggregated_signal (ранг по тикерам)
aggregated_signal + portfolio_snapshot -> risk_decision (портфель long/short)
risk_decision -> order_request -> execution_report
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
