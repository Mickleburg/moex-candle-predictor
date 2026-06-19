# Execution Source

Implemented (paper-first; live gated off by default). See `../README.md` for the full design,
broker choice (T-Invest, sandbox-first), modes, and acceptance mapping.

| Module | Role |
|--------|------|
| `config.py` | `Mode` (dry-run/paper/live), lot sizes, `SanityLimits`, the live gate (`live_enabled`) |
| `trading_calendar.py` | weekend-skip + injectable RU-holiday seam (backend owns the canonical calendar) |
| `reconcile.py` | `risk_book` + prices + current positions → delta **LIMIT** orders (lot-rounded, capped) |
| `discipline.py` | H9 **−12 enter / −2 exit** guard (critical if held into the ex-gap) |
| `audit.py` | append-only JSONL audit log (`var/audit/`, gitignored) |
| `engine.py` | cycle orchestration, duplicate ledger, **kill-switch**, season replay |
| `cli.py` | `dry-run` / `paper-season` entry points |
| `brokers/` | `BrokerAdapter` ABC, `DryRunBroker`, `PaperBroker` (sim), `TInvestBroker` (sandbox/live, gated) |

Entry points: `execution/scripts/demo_dry_run.py`, `execution/scripts/demo_paper_season.py`,
`python -m execution.src.cli`.
