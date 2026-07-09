# Execution Source

Implemented (paper-first; live gated off by default). See `../README.md` for the full design,
broker choice (T-Invest, sandbox-first), modes, and acceptance mapping.

| Module | Role |
|--------|------|
| `config.py` | `Mode` (dry-run/paper/live), lot sizes, `SanityLimits`, the live gate, env-overridable runtime dirs |
| `trading_calendar.py` | `default_trading_calendar()` → backend RU-holiday canon (else weekday fallback); `np.busday_count` counting |
| `instruments.py` | `load_lot_sizes()` / `load_figi_map()` — backend-first instrument metadata, fall back to defaults |
| `reconcile.py` | `risk_book` + prices + current positions → delta **LIMIT** orders (lot-rounded, capped) |
| `discipline.py` | H9 **−12 enter / −2 exit** guard (critical if held into the ex-gap) |
| `audit.py` | append-only JSONL audit log (`var/audit/`, gitignored) |
| `engine.py` | cycle orchestration, `reconcile_and_execute` (orchestrator envelope), duplicate ledger, **kill-switch**, season replay |
| `cli.py` | `serve` (orchestrator seam: stdin→stdout JSON) / `dry-run` / `paper-season` / `kill` / `unkill` |
| `brokers/` | `BrokerAdapter` ABC, `DryRunBroker`, `PaperBroker` (sim), `TInvestBroker` (sandbox/live, gated) |

Orchestrator seam (agent step 6): `python -m execution.src.cli serve --mode <mode>` — request envelope
on stdin, `{orders, reports, positions, rejected}` on stdout. Entry points:
`execution/scripts/demo_serve.py`, `demo_dry_run.py`, `demo_paper_season.py`.
