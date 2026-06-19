# Demo branch final audit

## Цель

Финально сверить ветку `demo` как стабильную архитектурную базу перед отделением новой ветки для дальнейших ML-экспериментов. Этот audit не включает новые research-прогоны, final test evaluation, production artifact bundle или реализацию торговых блоков.

## Текущая ветка и commits

Проверенная ветка:

```text
demo
```

Последние commits на момент audit:

```text
5438794 Validate demo architecture contracts
ad76c25 Scaffold trading agent demo architecture
571c115 Record focused triple-barrier validation result
49045b9 Add strict triple-barrier research pipeline
8982314 Add feature ablation and target grid research
```

## Структура проекта

Проверены основные директории:

- `backend/`;
- `ml/`;
- `llm/`;
- `aggregator/`;
- `risk_manager/`;
- `execution/`;
- `agent/`;
- `contracts/`;
- `config/`;
- `scripts/`.

Проверены обязательные README:

- `README.md`;
- `backend/README.md`;
- `ml/README.md`;
- `ml/docs/README.md`;
- `ml/docs/research/README.md`;
- `llm/README.md`;
- `aggregator/README.md`;
- `risk_manager/README.md`;
- `execution/README.md`;
- `agent/README.md`;
- `contracts/README.md`.

SBER H1 research reports перенесены в `ml/docs/research/`. В корневом `docs/` нет старых `sber_h1_*.md`.

## Блоки

- `backend` - существующий Go data/backend блок для свечей и backend API.
- `ml` - рабочий Python ML research/inference блок.
- `llm` - scaffold будущего LLM technical-analysis блока.
- `aggregator` - scaffold будущего late-fusion блока.
- `risk_manager` - scaffold будущего блока риск-ограничений.
- `execution` - scaffold будущего dry-run/paper/live execution adapter.
- `agent` - scaffold будущего orchestrator.
- `contracts` - JSON Schema и согласованные examples между блоками.
- `config` - demo-supported ticker config.

## Текущий ML baseline

Актуальный лучший validation-only candidate:

```text
target:       triple_barrier:h3:w12:up1.25:down1.25
features:     continuous_regime
model:        extra_trees:depth=none:leaf=20:maxfeat=sqrt
class_weight: none
mean macro-F1: 0.4695
worst fold:    0.4548
BUY F1:        0.4064
SELL F1:       0.4387
HOLD F1:       0.5632
action_rate:   0.6725
```

Ограничения:

- результат validation-only;
- это не production artifact;
- test split не должен использоваться для нового tuning;
- перед frozen candidate нужны seed robustness и отдельный frozen evaluation protocol.

## Contracts validation

Проверены contracts и examples:

- `candle_batch`;
- `market_snapshot`;
- `portfolio_snapshot`;
- `ml_prediction`;
- `llm_analysis`;
- `aggregated_signal`;
- `risk_decision`;
- `order_request`;
- `execution_report`;
- `agent_cycle_result`.

`scripts/validate_contracts.py` проверяет:

- JSON syntax для schemas и examples;
- базовую форму JSON Schema;
- probability sums для `buy/hold/sell`;
- ML -> LLM -> aggregator -> risk -> execution consistency;
- supported tickers.

`jsonschema` является optional dependency. Если пакет не установлен, schema-vs-example validation пропускается с явным сообщением, а ручные cross-contract checks продолжают выполняться.

## Supported tickers

Проверены demo-supported тикеры в `config/supported_tickers.yaml`:

- `SBER`;
- `GAZP`;
- `LKOH`.

Для каждого тикера ожидаются поля:

- `ticker`;
- `name`;
- `market`;
- `board`;
- `enabled`;
- `timeframes`.

## ML compatibility

Проверяемые команды:

```powershell
python -m compileall -q ml\src ml\scripts ml\test_smoke.py
python -m compileall -q ml\scripts
python ml\test_smoke.py
```

Ожидаемый статус для ветки `demo`: pass.

## Backend compatibility

Проверяемая команда из `backend/`:

```powershell
go test ./...
```

Ожидаемый статус для ветки `demo`: pass, если Go toolchain доступен.

## Safety status

- Реальная торговля по умолчанию запрещена.
- Первый execution режим должен быть dry-run/paper.
- LLM не торгует и не отправляет orders.
- Aggregator не торгует и не применяет финальные risk limits.
- Risk manager сильнее ML/LLM/aggregator и может заблокировать любой сигнал.
- Execution должен принимать только approved order intent.
- Текущий ML best не является production artifact.
- Test split нельзя использовать для research tuning.

## Что не реализовано

- LLM client;
- aggregator logic;
- risk manager implementation;
- execution implementation;
- broker/MOEX API integration;
- paper/live trading implementation;
- production ML artifact bundle;
- new ML research tuning.

## Известные ограничения

- `jsonschema` optional: без него выполняются JSON/cross-contract checks, но не full schema-vs-example validation.
- Нет live trading.
- Нет production artifact.
- Нет нового test tuning.
- Current best ML candidate требует seed robustness.

## Готовность к отделению новой ветки

Ветка `demo` готова быть базовой архитектурной точкой для ответвления новой ML-ветки при условии, что финальные проверки проходят:

```powershell
python -m compileall -q ml\src ml\scripts ml\test_smoke.py
python ml\test_smoke.py
python scripts\validate_contracts.py
git diff --check
```

## Рекомендуемая следующая ветка

```text
ml-triple-barrier-seed-robustness
```

Альтернатива для более широкой работы:

```text
ml-experiments-after-demo
```
