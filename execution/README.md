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

## Shadow gate — which book each mode trades

The risk_manager splits the `risk_book` by a shadow gate (invariants #9/#4): proven LIVE sleeves go to
`net_positions` (+ live `hedge`); gated-out, paper-only sleeves go to `shadow_positions` (+
`shadow_hedge`) with **zero live capital**. Execution honours that split **by mode**:

```
dry-run / paper -> effective book = net_positions ∪ shadow_positions  (+ hedge ∪ shadow_hedge)
                   -> paper-trades the shadow book so the forward-shadow track accrues through the
                      real execution path (the point of the paper track)
live            -> net_positions + live hedge ONLY
                   -> an all-shadow book (e.g. H9 while is_production=false) places ZERO real orders
```

So while H9 is `is_production=false` its whole book is shadow: paper/dry-run reconcile and paper-fill
it; live places nothing. Names appearing in both the live and shadow books net by ticker.

## Orchestrator integration — the `serve` seam (agent step 6)

The agent invokes execution exactly the way it invokes the ML/risk_manager CLIs
(`agent/src/adapters/live.py::LiveExecution`): it appends `--mode <mode>`, feeds a request envelope as
JSON on **stdin**, and parses the result JSON from **stdout**. Only `serve` writes to stdout, and it
writes JSON *only* (the human summary goes to stderr), so `json.loads(proc.stdout)` is safe.

```
stdin  request = {risk_book, positions, prices, capital, mode, trade_date, phase, [anchors]}
stdout result  = {orders, reports, positions, rejected, halted, is_production}
                  orders   -> order_request[]      reports -> execution_report[]
                  positions-> book after fills      rejected-> [{ticker, reason}]
```

Wire it into `agent/config/agent_config.json` (then flip `blocks.execution.mode` off `mock`):

```json
"execution": {
  "mode": "live",
  "command": ["ml/.venv-win/Scripts/python.exe", "-m", "execution.src.cli", "serve"]
}
```

On the Linux VDS the command is the same with that environment's interpreter, e.g.
`["python", "-m", "execution.src.cli", "serve"]` (run from the repo root). The agent appends
`--mode paper|live`; execution maps it to its mode and the live gate still applies. Manual check:

```powershell
Get-Content execution/examples/serve_request.example.json -Raw `
  | & "ml\.venv-win\Scripts\python.exe" -m execution.src.cli serve --mode paper
```

Runtime dirs are env-overridable (`EXECUTION_STATE_DIR`, `EXECUTION_AUDIT_DIR`) so the VDS/systemd can
place the dedupe ledger + audit off the repo; the kill-switch (`KILL` file) lives in the state dir, so
`serve`, `kill`, and `unkill` must share it.

## What it does

1. **Reconciliation** (`src/reconcile.py`) — the effective-book (per the shadow gate above) weights ×
   book capital ÷ reference price → target lots, **rounded DOWN** to MOEX round lots, per-name
   sanity-capped, diffed against current holdings. Only non-zero diffs become orders. Names that
   **drop out** of the book are flattened
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

## Trading calendar — the backend RU-holiday canon

The −12/−2 discipline counts trading days on the **same canon as the sleeve and monitor**:
`default_trading_calendar()` delegates to `backend.trading_calendar` (RU holidays + the real IMOEX
panel overlay) when importable, so the timing cannot drift on a private holiday list across the
May–July record-date cluster. `trading_days_between` uses the `np.busday_count` `[start, end)`
convention the sleeve uses (`td = trading_days_between(as_of, record_date)`). If backend is not
importable (isolated env), execution falls back to a weekday-only `TradingCalendar` with the same
counting convention; you can still inject holidays explicitly (`TradingCalendar(holidays=…)`).
`active_calendar_source()` reports which is live (`backend_canon` vs `weekday_fallback`).

## Instrument reference data (lot sizes / FIGI) — backend-first

`src/instruments.py::load_lot_sizes()` / `load_figi_map()` prefer a backend source and fall back to
`config.DEFAULT_LOT_SIZES` (defaults to confirm) + an empty FIGI map. Execution is not the source of
truth for instrument data. **Open dependency:** `backend.universe.Instrument` currently carries no
`lot_size` / `figi`; when backend exposes them (a `backend.instruments.lot_sizes()/figi_map()` lookup,
or a `lot_size` field on the universe), the loaders pick them up with no change here. Until then the
live T-Invest path must be handed a FIGI map explicitly — it refuses to guess. Sector/market hedge
legs (`MOEX*`) are not directly lot-traded; in live they are worked via index futures/ETF (lot=1 here
is a paper placeholder).

## Usage

```powershell
$PY = "ml\.venv-win\Scripts\python.exe"

# Orchestrator seam: request envelope on stdin -> result JSON on stdout
& $PY execution/scripts/demo_serve.py
Get-Content execution/examples/serve_request.example.json -Raw `
  | & $PY -m execution.src.cli serve --mode paper

# Dry-run the example risk_book into delta orders (sends nothing)
& $PY execution/scripts/demo_dry_run.py
& $PY -m execution.src.cli dry-run --risk-book contracts/examples/risk_book.example.json `
      --prices execution/examples/prices.example.json

# Replay the illustrative H9 run-up season through the paper simulator
& $PY execution/scripts/demo_paper_season.py

# Kill-switch (engage / clear)
& $PY -m execution.src.cli kill --reason "manual stop"
& $PY -m execution.src.cli unkill

# Tests
& $PY -m pytest execution/tests -q
```

## Acceptance (mapped)

- ✅ documented CLI command emits `execution_report`s from the `risk_book` example: `serve` (stdin
  envelope) returns `{orders, reports, …}` — see `demo_serve.py` / `test_serve.py`.
- ✅ dry-run prints correct delta orders from the `risk_book` example, sends nothing.
- ✅ −12/−2 discipline counts on the backend RU-holiday canon (`test_calendar_backend.py`).
- ✅ paper (sim/sandbox) executes entry/exit on the −12/−2 discipline; full audit log; kill-switch
  cancels all + halts.
- ✅ duplicate protection blocks a second order for the same intent (within and across processes).
- ✅ live path exists but is **blocked** without the explicit flag; `is_production=false` on artifacts.
- ✅ paper season replay matches the sleeve's hold window (`test_paper_season.py`).
- ✅ `python -m pytest ml/test_smoke.py` green; contract validation green (no contract changes).

## Layout

```
execution/
  README.md            this file
  .env.example         secrets/flags template (copy to .env; .env is gitignored)
  src/
    config.py          modes, lot sizes, sanity limits, live gate, env-overridable runtime dirs
    trading_calendar.py backend RU-holiday canon (default) + weekday fallback, np.busday_count count
    instruments.py     lot-size / FIGI loaders (backend-first, fall back to defaults)
    reconcile.py       risk_book + prices + positions -> delta LIMIT orders
    discipline.py      H9 -12/-2 entry/exit guard
    audit.py           append-only JSONL audit log
    engine.py          cycle orchestration, reconcile_and_execute, dedupe, kill-switch, season replay
    cli.py             serve (orchestrator seam) / dry-run / paper-season / kill / unkill
    brokers/           base ABC, DryRun, PaperBroker (sim), TInvestBroker (sandbox/live, gated)
  examples/            prices, illustrative season, serve request envelope
  scripts/             demo_serve.py, demo_dry_run.py, demo_paper_season.py
  tests/               reconcile, discipline, protections, season, serve, contract conformance, calendar
  var/                 runtime audit/state (gitignored; override via EXECUTION_STATE_DIR/AUDIT_DIR)
```

## Discipline

Commit only `execution/…` + its tests. Live only after the shadow gate + team sign-off + the explicit
enable flag. Broker secrets only in `.env` / a secret store — never in git.
