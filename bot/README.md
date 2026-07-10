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

Admin-only management commands (visible in `/help` only to admins):

- `/users` — show the allowlist: admins (👑, immutable), env seed, managed (note + who added)
- `/allow <chat_id> [note]` — grant read access at runtime (no restart)
- `/deny <chat_id>` — revoke a **managed** id (refuses admins / env-seed ids)

live and shadow capital are always rendered as separate, labelled sections — never conflated.

## Access control (two tiers, fail-closed)

- **admin** — `BOT_ADMIN_CHAT_IDS` (env bootstrap, the root of trust). May run the management
  commands and **can never be removed via the bot** (fail-safe against self-lockout).
- **allowed** — may run read commands. Effective set = **admins ∪ `BOT_ALLOWED_CHAT_IDS` (env
  seed) ∪ managed store**. The managed store is the bot's OWN file `data/bot/allowlist.json`
  (gitignored, atomic writes) — *not* the agent DB, which the bot only ever opens `mode=ro`.
  `/allow` / `/deny` edit it and take effect **immediately** (the set is read dynamically).
- A non-whitelisted user who messages the bot gets a short reply with **their own chat id** (not
  silence) so requesting access is self-service; admins also get a best-effort notification.
- **Fail-closed:** with no admins AND no allowed ids the bot **refuses to start**.
- **Token from the environment only.** `TELEGRAM_BOT_TOKEN`, never in git. See `.env.example`.
- **RU-blocked host?** Where `api.telegram.org` is unreachable (e.g. a Russian VDS behind RKN), set
  `TELEGRAM_PROXY_URL=http://<proxy-ip>:<port>` to an out-of-region HTTP proxy locked to the host's
  IP. Both the poller and `getUpdates` route through it (`Application.builder().proxy()`); the agent
  notifier proxies Telegram only (ISS stays direct). Leave blank if Telegram is directly reachable.

## Coordination with the agent notifier (one token)

The agent's `agent/src/notifier.py` only **pushes** (`sendMessage`). This bot is the single
**poller** (`getUpdates` / long-poll via `Application.run_polling`). Telegram allows only one
getUpdates consumer per token — **the bot is that consumer; the agent only sends, so there is no
conflict.** Never start a second poller on the same token.

## Run (local)

```powershell
# .env at repo root (gitignored): TELEGRAM_BOT_TOKEN=...  BOT_ADMIN_CHAT_IDS=<your id>
& "ml\.venv-win\Scripts\python.exe" -m bot
```

`python -m bot` loads `.env` (local convenience; on the VDS the process manager loads the env),
reuses the agent config for paths/universe, then long-polls. On startup it registers the command
menu via `set_my_commands` (read commands for everyone; admin commands scoped to admin chats) —
best-effort, so a proxy/network hiccup logs and continues rather than blocking start. VDS deploy
(Docker Compose / systemd) is live — see `infra/README.md`.

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
