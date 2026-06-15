# Agent Orchestrator Block

`agent/` — scaffold будущего orchestrator (Python). Запускает один цикл V2.

Планируемый цикл (архитектура V2 — кросс-секция + ранняя фьюжн):

1. прочитать вселенную тикеров (`config/supported_tickers.yaml`);
2. для всей вселенной получить свежие свечи + market context (backend);
3. получить новостные **признаки** от LLM-блока (не buy/hold/sell);
4. собрать `[цена ⊕ новости]` и получить **кросс-секционный ранг** от ML-блока
   (одна решающая модель, ранняя фьюжн — отдельного aggregator-шага больше нет);
5. передать ранг в `risk_manager` → портфель long top-k / short bottom-k, маркет-нейтрал;
6. передать approved orders в `execution`;
7. записать `agent_cycle_result`.

Агент не реализован. Live trading запрещён. Первый целевой режим — paper trading.
