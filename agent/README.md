# Agent Orchestrator Block

`agent/` - scaffold будущего orchestrator.

Планируемый цикл:

1. прочитать `config/supported_tickers.yaml`;
2. для каждого тикера запросить fresh candles;
3. получить ML prediction;
4. получить LLM analysis;
5. объединить сигналы через aggregator;
6. запросить risk approval;
7. выбрать одну или несколько лучших сделок по risk-adjusted score;
8. передать approved orders в execution;
9. записать `agent_cycle_result`.

Агент не реализован. Live trading запрещен. Первый целевой режим - paper trading.
