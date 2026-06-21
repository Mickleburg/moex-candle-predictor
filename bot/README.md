# `bot/` — Telegram monitoring bot (read-only observer)

Interactive Telegram bot to **observe** the MOEX V3 trading agent on demand: positions, prices,
P&L, gate/regime/data status, plus readable digests. **Read-only by default — it never trades and
exposes no control actions.** It only *reads* the state the agent already maintains.

## What it reads (it never recomputes or hits the network)

| Source | Path | Used by |
|--------|------|---------|
| Agent state store (SQLite, **opened `mode=ro`**) | `data/agent/state.sqlite` (`agent_config.json → paths.state_db`) | `/status` `/positions` `/pnl` `/cycle` `/gate` |
| Data-integrity report | `data/reports/data_integrity_status.json` | `/integrity` |
| Shadow-gate verdict | `data/reports/h9_shadow_pnl.txt` | `/gate` |
| Candle store (last bar) | `data/raw/*.parquet` via `backend.store` | `/prices` |

Schema/read semantics mirror `agent/src/state_store.py` (the write owner) — see
`bot/src/datasource.py`. Every accessor degrades gracefully: missing DB/report → "no data", never a
crash.

## Commands (all read-only)

- `/status` — mode, kill-switch, last cycle, live/shadow gross (directional shown separately from hedge legs, so it never reads as a misleading >100% of capital)
- `/positions` — live + shadow book, by name (lots / weight / sector)
- `/pnl` — P&L by sleeve, **live separated from shadow**
- `/prices [TICKERS]` — last close (defaults to the universe)
- `/gate` — shadow gate: `is_production`, MET/NOT MET, forward P&L
- `/shadowlog [N]` — last N forward-shadow cycles (`data/agent/shadow_pnl.jsonl`): date, sleeves, shadow P&L per sleeve (default 5)
- `/cycle` — last EOD result: orders, binding limits, alerts
- `/integrity` — data gate HALT/OK + reasons
- `/help`

live and shadow capital are always rendered as separate, labelled sections — never conflated.

## Security (fail-closed)

- **Chat-id whitelist.** `BOT_ALLOWED_CHAT_IDS` (comma-separated owner ids). Any update from a
  non-whitelisted chat is ignored + logged. An **empty whitelist makes the bot refuse to start.**
- **Token from the environment only.** `TELEGRAM_BOT_TOKEN`, never in git. See `.env.example`.

## Coordination with the agent notifier (one token)

The agent's `agent/src/notifier.py` only **pushes** (`sendMessage`). This bot is the single
**poller** (`getUpdates` / long-poll via `Application.run_polling`). Telegram allows only one
getUpdates consumer per token — **the bot is that consumer; the agent only sends, so there is no
conflict.** Never start a second poller on the same token.

## Run (local)

```powershell
# .env at repo root (gitignored): TELEGRAM_BOT_TOKEN=...  BOT_ALLOWED_CHAT_IDS=<your id>
& "ml\.venv-win\Scripts\python.exe" -m bot
```

`python -m bot` loads `.env` (local convenience; on the VDS the process manager loads the env),
reuses the agent config for paths/universe, then long-polls. VDS deploy (systemd/docker) is a
**separate, later** step coordinated with infra.

## Control actions — intentionally OUT of v1

The bot is read-only. A kill-switch toggle was deliberately **not** implemented: if ever added it
must be whitelist-only with double confirmation, writing `state_store.set_kill_switch`. Placing real
orders via the bot is **never** allowed.

## Tests

```powershell
& "ml\.venv-win\Scripts\python.exe" -m pytest bot/tests -q
```

Logic (config/whitelist, datasource, monitor, formatters) is library-agnostic and tested offline
against a seeded `state.sqlite`; `python-telegram-bot` is imported lazily in `bot/src/app.py` only,
so the tests need no network and no live token. `is_production=false` throughout.
