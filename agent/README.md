# Agent Orchestrator Block (V3)

`agent/` is the **glue**: the autonomous daily cycle that wires the V3 blocks together around
the H9 dividend run-up sleeve and keeps it alive on a VDS. It owns the operational loop, the
durable state store, the scheduler, and the alert/monitoring plane. It calls every other block
**only through JSON contracts / public APIs** — it never reaches into their internals.

> Supersedes the old V2 cross-sectional description. V2 (cross-section + early fusion) is
> closed; the profitable path is a portfolio of weakly-correlated **sleeves** + a risk layer
> (`docs/ARCHITECTURE_V3.md`). The first robust edge is **H9** (pre-ex dividend run-up).

## What "autonomous" means (the five properties)

The agent itself: **(1)** collects data · **(2)** monitors · **(3)** reacts to change ·
**(4)** updates the knowledge base · **(5)** manages the portfolio. The daily cycle below
realises each one.

## Daily cycle (state machine)

**EOD (~19:05 MSK, trading days only):**

| # | Step | Block (adapter) | Property |
|---|------|-----------------|----------|
| 1 | ingest today's candles + market context | backend | (1) |
| 3 | data-integrity gate — **HALT ⇒ do not trade** | backend | (2) |
| 4 | ML sleeve `build_sleeve_signal(as_of)` → `sleeve_signal` | ml | (5) |
| 5 | combiner: net × vol-target × **H5 regime gate** × limits × hedge → `risk_book` | risk_manager | (5) |
| 6 | reconcile vs current book → LIMIT delta-orders (paper) | execution | (5) |
| 7 | persist book + orders + **per-sleeve P&L attribution** + shadow log | agent/state | (2)(4) |
| 8 | alert digest (entries/exits, gate, P&L, data failures) | agent/monitoring | (2) |

**Pre-open (~09:30 MSK):** kill-switch check, overnight gap/HALT check, confirm/cancel resting
limit orders.

**Reactions (3):** data HALT → no trading · H5 regime gate → the combiner already cut gross and
the agent executes that smaller book · a new ex-date → enters via the sleeve at EOD ·
kill-switch → trading stops, monitoring continues.

Every cycle is **idempotent** per `(trade_date, phase)` and **recoverable after restart** — all
state lives in the SQLite store.

## Module map

```
agent/src/
  config.py            JSON config + env overlay (no secrets in the file)
  contracts.py         validate every block-seam payload against contracts/
  trading_calendar.py  RU-holiday-aware days; uses the backend calendar, else a vendored fallback
  state_store.py       SQLite: cycle_runs (idempotency), positions, orders, executions, pnl, kv
  notifier.py          Telegram (stdlib urllib) | stdout fallback
  pnl.py               per-sleeve P&L attribution + shadow-log writer
  orchestrator.py      the EOD + pre-open state machine
  scheduler.py         APScheduler daemon (EOD/pre-open/dead-man's-switch), TZ Europe/Moscow
  cli.py               run-eod | run-preopen | status | kill-switch | init-db | scheduler
  adapters/            block seams: mock (default) + live impls + registry
config/agent_config.json   default config (paper + mock, runs end-to-end with no other block)
tests/                 full-cycle, idempotency, restart, HALT, regime-cut, kill-switch
```

The orchestrator **core is stdlib-only**; heavy deps (pandas, the ml/risk_manager packages) are
imported lazily inside the `live` adapters, so the mock cycle and the tests run with stdlib alone.

## Block adapters (mock + live)

Each block sits behind an interface with a `mock` and a `live` implementation, wired by
`block_mode` (and per-block `blocks.<name>.mode`). Mock is the default so the full cycle runs
today; flip to live as each block is ready.

| Block | live path | mock path |
|-------|-----------|-----------|
| backend | `backend.ingest`/`backend.integrity`/`backend.store` in-process (or CLI) | healthy/HALT-injectable stub + synthetic prices |
| sleeve (ml) | `ml…dividend_sleeve.build_sleeve_signal` | canned long book + hedge rec |
| combiner (risk_manager) | `risk_manager.src.combine` + ml `risk_analytics` | self-contained netting + regime/vol knobs |
| execution | execution-block CLI (paper→live) | deterministic paper broker (LIMIT, lot-rounding, fills) |

## Usage

```bash
python -m agent.src.cli run-eod --force          # one EOD cycle now (paper, mock blocks)
python -m agent.src.cli run-preopen --force      # one pre-open check
python -m agent.src.cli status                   # mode, kill-switch, book, open orders, P&L
python -m agent.src.cli kill-switch on|off        # emergency stop
python -m agent.src.cli scheduler                # the long-lived daemon (needs APScheduler)
```

Tests: `& "ml\.venv-win\Scripts\python.exe" -m pytest agent/tests -q`.

## Invariants

`is_production=false` flows through every artifact until the forward-shadow gate + team sign-off.
**Live is double-gated**: `AGENT_ENABLE_LIVE=true` AND `mode=live` — either alone stays paper
(paper-first). Secrets only in `.env` / `EnvironmentFile`, never in git. Deploy: `infra/README.md`.
Source of truth: `docs/VDS_AUTONOMOUS_PLAN.md`, `docs/ARCHITECTURE_V3.md`, `docs/RESEARCH_HYPOTHESES.md`.
