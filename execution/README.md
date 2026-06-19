# Execution Block — broker adapter (paper→live) + H9 order discipline

Step **6** of the VDS daily cycle (`docs/VDS_AUTONOMOUS_PLAN.md`): take the target book from the
risk_manager (`risk_book`), reconcile it against current positions, and emit lot-rounded **LIMIT**
delta orders under the H9 dividend run-up discipline. **No real orders by default** — `dry-run` and
`paper` first; `live` is gated behind an explicit flag and stays off until the forward-shadow gate +
team sign-off. `is_production=false` is stamped on every artifact until then.

## Broker choice — T-Invest (Tinkoff Invest), sandbox-first

**Decision: T-Invest API.** Rationale (per `docs/CHAT_TASK_execution.md` candidate list):

| Candidate | Paper path | Verdict |
|-----------|-----------|---------|
| **T-Invest** | First-class **sandbox** mirroring the production order API | **Chosen** — cleanest paper→live promotion; one REST gateway, token auth |
| Finam TradeAPI | No true sandbox | Heavier paper story |
| ALOR OpenAPI | Limited sandbox | Viable, more setup |
| QUIK connector | Terminal-bound (LUA/DDE) | Not headless-VDS friendly |

The sandbox lets paper trading run the *real* order path with fake money, so a paper season can be
reconciled against the sleeve backtest before any capital is at risk. The adapter
(`src/brokers/tinvest.py`) talks the public REST gateway with the **stdlib only** (no extra deps) and
constructs lazily — the block stays importable without credentials or network. Default paper backend
is an internal deterministic simulator (`PaperBroker`) so tests/CI are hermetic; switch to the
T-Invest sandbox with `broker_backend="tinvest"`.

## Modes (escalating, live off by default)

```
dry-run  ── reconcile + print delta orders, contact no broker
paper    ── execute against the internal simulator (default) or the T-Invest sandbox
live     ── real orders — REFUSED unless ALL of:
            ExecutionConfig.allow_live=True  AND  env EXECUTION_ALLOW_LIVE=1  AND  broker_backend="tinvest"
```

There is no accidental live path: `make_broker` raises `PermissionError` if any gate is unmet.

## What it does

1. **Reconciliation** (`src/reconcile.py`) — `risk_book` weights × book capital ÷ reference price →
   target lots, **rounded DOWN** to MOEX round lots, per-name sanity-capped, diffed against current
   holdings. Only non-zero diffs become orders. Names that **drop out** of the book are flattened
   (that is the exit). Limit price = reference close, so a paper replay reproduces the close-to-close
   sleeve sim. Output conforms to `contracts/order_request.schema.json`.
2. **H9 discipline guard** (`src/discipline.py`) — given a dividend anchor (record/ex date) per name,
   verifies a name is held only when `exit_offset < td ≤ entry_offset` (default **−12 enter / −2
   exit**, counted on the trading calendar). Holding into the ex-gap (`td ≤ 2`) is **critical** and
   halts the cycle; early entry is a warning. Anchors are optional (names without one aren't checked).
3. **Protections** (`src/engine.py`) — duplicate-order protection (idempotent `client_order_id`
   ledger), **kill-switch** (`engine.kill()` cancels everything + halts; persisted as a `KILL` file),
   per-order/fill **audit log** (append-only JSONL), per-name sanity limits (max lots / max notional).
4. **Weekend skip** — MOEX weekend sessions have no edge; the engine refuses to trade on a
   non-trading-day `as_of`.
5. **Paper↔sim reconciliation** — `engine.run_season()` replays a sequence of daily books through the
   simulator; the holdings trace matches the sleeve's −12/−2 window (one entry, no churn, one exit).

## Trading calendar (seam, not a duplicate)

The canonical **RU-holiday** trading calendar is owned by the backend/data block (and the ML sleeve).
`src/trading_calendar.py` provides only what execution needs locally — the **weekend-skip** policy and
trading-day arithmetic — over an **injected** holiday set / predicate. In production the orchestrator
passes the backend calendar in (`holidays=…` or `is_trading_day=…`); the default is weekday-only,
which is already correct except across multi-day holiday clusters (May/June).

## Instrument reference data (seam)

Lot sizes (`src/config.py::DEFAULT_LOT_SIZES`) and FIGI mapping are **defaults to be confirmed /
overridden from backend instrument metadata** (MOEX ISS securities table) — execution is not the
source of truth for instrument data. Sector/market hedge legs (`MOEX*`) are not directly lot-traded;
in live they are worked via index futures/ETF (lot=1 here is a paper placeholder).

## Usage

```powershell
$PY = "ml\.venv-win\Scripts\python.exe"

# Dry-run the example risk_book into delta orders (sends nothing)
& $PY execution/scripts/demo_dry_run.py
& $PY -m execution.src.cli dry-run --risk-book contracts/examples/risk_book.example.json `
      --prices execution/examples/prices.example.json

# Replay the illustrative H9 run-up season through the paper simulator
& $PY execution/scripts/demo_paper_season.py

# Tests
& $PY -m pytest execution/tests -q
```

## Acceptance (mapped)

- ✅ dry-run prints correct delta orders from the `risk_book` example, sends nothing.
- ✅ paper (sim/sandbox) executes entry/exit on the −12/−2 discipline; full audit log; kill-switch
  cancels all + halts.
- ✅ duplicate protection blocks a second order for the same intent.
- ✅ live path exists but is **blocked** without the explicit flag; `is_production=false` on artifacts.
- ✅ paper season replay matches the sleeve's hold window (`test_paper_season.py`).
- ✅ `python -m pytest ml/test_smoke.py` green; contract validation green (no contract changes).

## Layout

```
execution/
  README.md            this file
  .env.example         secrets/flags template (copy to .env; .env is gitignored)
  src/
    config.py          modes, lot sizes, sanity limits, live gate
    trading_calendar.py weekend-skip + injectable RU-holiday seam
    reconcile.py       risk_book + prices + positions -> delta LIMIT orders
    discipline.py      H9 -12/-2 entry/exit guard
    audit.py           append-only JSONL audit log
    engine.py          cycle orchestration, dedupe, kill-switch, season replay
    cli.py             dry-run / paper-season CLI
    brokers/           base ABC, DryRun, PaperBroker (sim), TInvestBroker (sandbox/live, gated)
  examples/            prices + illustrative season
  scripts/             demo_dry_run.py, demo_paper_season.py
  tests/               reconcile, discipline, protections, season, contract conformance, calendar
  var/                 runtime audit/state (gitignored)
```

## Discipline

Commit only `execution/…` + its tests. Live only after the shadow gate + team sign-off + the explicit
enable flag. Broker secrets only in `.env` / a secret store — never in git.
