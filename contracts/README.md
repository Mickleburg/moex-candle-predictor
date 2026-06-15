# Shared Contracts

`contracts/` содержит JSON Schema для блоков торгового агента.

> **V2 контракты ЗАФИКСИРОВАНЫ (2026-06-15).** Переопределены под кросс-секцию + раннюю фьюжн.
> Проверка: `python scripts/validate_contracts.py` (с `jsonschema` — валидация примеров против схем).
> Блок `aggregator` удалён (ранняя фьюжн). См. `docs/ARCHITECTURE_V2.md`.

## Статус контрактов (V2)

| Контракт | Статус | Роль |
|----------|--------|------|
| `candle_batch` | без изменений | вход ML/бэкенда: OHLCV по тикеру |
| `market_snapshot` | без изменений | market context (индексы/индикаторы) |
| `llm_analysis` | **переопределён** | НОВОСТНЫЕ ПРИЗНАКИ по тикеру (sentiment/event/impact/novelty/embedding), НЕ buy/hold/sell |
| `feature_bundle` | **новый** | `[quant ⊕ news]` матрица по вселенной — вход решающей модели (ранняя фьюжн) |
| `aggregated_signal` | **переопределён** | кросс-секционный РАНГ вселенной (long top-k / short bottom-k), выход решающей модели |
| `risk_decision` | +`position_side` (LONG/SHORT/FLAT) | портфельное решение по тикеру (маркет-нейтрал) |
| `ml_prediction` | без изменений | пер-тикерный buy/hold/sell прогноз (V1, всё ещё обслуживается ML-сервисом; для диагностики/одиночных бумаг) |
| `order_request`, `execution_report`, `portfolio_snapshot`, `agent_cycle_result` | без изменений | исполнение/учёт |

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
