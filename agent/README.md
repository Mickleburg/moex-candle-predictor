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
| 2 | refresh the upcoming-dividend feed (best-effort; failure alerts, never blocks) | llm | (1)(4) |
| 3 | data-integrity gate — **HALT ⇒ do not trade** | backend | (2) |
| 4 | ML sleeve `predict_dividend_sleeve.py --out -` → `sleeve_signal` | ml | (5) |
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

## Dead-man's-switch

Interval job (`deadman_check_minutes`, default 30) that asks: *did the most recent DUE EOD complete?*
The reference date is the last trading day whose EOD time + 15 min has passed — so before ~19:20 MSK it
still refers to **yesterday**. Decision logic is the pure `deadman_verdict()` in `scheduler.py`
(unit-tested offline); `deadman_tick()` does the read → verdict → send → persist.

| Cycle status | Behaviour |
|--------------|-----------|
| `completed` / `halted` / `killed` | healthy — no alert, **and the dedup flag is cleared** so a future failure alerts at once |
| `running`, younger than `deadman_running_grace_minutes` (default 90) | in flight on a slow box — **silent** (a real EOD takes longer than the 15-min reference margin) |
| `running`, older than the grace | died mid-cycle — alert (a distinct *stuck* message) |
| `failed` / missing | alert |

**Deduplicated:** the same incident (same `reference_date:status`) re-alerts **at most once per
`deadman_repeat_hours`** (default 6). A *new* incident — different date or different status — alerts
immediately. A genuinely dead agent therefore keeps reminding; it just stops spamming every 30 min.
The alert is persisted as "sent" **only when delivery succeeds**, so a proxy/network failure never
arms the quiet window and swallows an alert nobody received. Flag lives in the `kv` table
(`deadman_last_alert`).

> Tuning: `deadman_running_grace_minutes` must exceed the real EOD duration on your box, or a healthy
> slow cycle is reported as stuck. Measure it from the `EOD cycle start` → `EOD cycle done` log pair.

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
| sleeve (ml) | `ml/scripts/predict_dividend_sleeve.py --out -` (subprocess seam — no pandas in the agent core) | canned long book + hedge rec |
| combiner (risk_manager) | `risk_manager.src.combine` + ml `risk_analytics` | self-contained netting + regime/vol knobs |
| execution | the real `execution.ExecutionEngine` in-process (discipline −12/−2, lot rounding, dup-ledger, audit, paper broker) | deterministic paper broker stub |

The default config runs **execution live (paper sim)** + the other blocks mock, so the cycle goes
through real execution-block code out of the box; flip the rest to `live` as data is seeded. The
trading calendar is a re-export of the canonical `backend.trading_calendar` (one source of truth).

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
