# Deploy runbook — MOEX autonomous agent + monitoring bot on a VDS (PAPER)

Bring the V3 stack up on a Linux VDS (Ubuntu LTS) so the orchestrator runs the EOD cycle **every
trading day** and accrues the forward-shadow track through the dividend season, with a Telegram
bot for on-demand monitoring. **Paper only — never live until sign-off** (see the fenced section
at the bottom).

> Decisions (the umbrella plan delegated them): **Docker Compose** primary one-command deploy +
> supervisor, **systemd** venv alternative; **in-process APScheduler** for EOD/pre-open +
> dead-man's-switch; **SQLite** state store; **Telegram** for both push alerts and the bot.

## Two services, one token

| Service | Process | Telegram role | Notes |
|---------|---------|---------------|-------|
| `agent` | `python -m agent.src.cli scheduler` | **sendMessage only** (push digests/HALT/dead-man) | APScheduler EOD 19:05 + pre-open 09:30 MSK |
| `bot`   | `python -m bot` | **the SINGLE getUpdates poller** (read-only) | reads the agent state DB `mode=ro` |

They are **separate processes/services** (two containers, or two systemd units) — not one event
loop — so the scheduler trigger and the long-poller never block each other. Telegram allows one
`getUpdates` consumer per token; the **bot is that consumer, the agent only pushes**, so a single
`TELEGRAM_BOT_TOKEN` is shared with **no conflict**. Never start a second poller on the token.

Both share the bind-mounted `data/` dir: the agent writes the SQLite store / cycle results /
shadow log; the bot reads them `mode=ro` and writes only its own `data/bot/allowlist.json`.

## 1. Provision the VDS (Ubuntu, root) — first byte

```bash
# fresh Ubuntu LTS, as root
apt-get update && apt-get install -y git ca-certificates curl
# Docker Engine + compose plugin (Option A)
install -m0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] \
https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) stable" \
  > /etc/apt/sources.list.d/docker.list
apt-get update && apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
timedatectl set-timezone Europe/Moscow

git clone <repo-url> /opt/moex-candle-predictor && cd /opt/moex-candle-predictor
```

## 2. Secrets — `/etc/moex-agent.env` (chmod 600, never from git)

Copy the template, fill it in, lock it down. Secrets live ONLY here (or the repo-root `.env` for
Docker `env_file`) — never committed.

```bash
cp .env.example /etc/moex-agent.env
chmod 600 /etc/moex-agent.env
$EDITOR /etc/moex-agent.env
```

Minimum for a paper deploy with monitoring:

```ini
AGENT_MODE=paper
AGENT_ALERT_CHANNEL=telegram
TELEGRAM_BOT_TOKEN=<from @BotFather>     # shared by agent (push) + bot (poll)
TELEGRAM_CHAT_ID=<your chat id>          # where the agent PUSHES alerts (@userinfobot)
BOT_ADMIN_CHAT_IDS=<your chat id>        # bot admin(s); without admins/allowed the bot fail-closes
# AGENT_ENABLE_LIVE / EXECUTION_ALLOW_LIVE stay OFF (see paper lock below)
```

For Docker, the same file at the repo root as `.env` (the compose `env_file`):
`cp /etc/moex-agent.env .env && chmod 600 .env`.

## 3a. Start — Docker Compose (recommended)

```bash
docker compose -f infra/docker-compose.yml up -d --build      # builds the image once, starts agent + bot
docker compose -f infra/docker-compose.yml ps                  # both Up
docker compose -f infra/docker-compose.yml logs -f agent bot   # tail
```

## 3b. Start — venv + systemd (alternative)

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
useradd --system --home /opt/moex-candle-predictor agent && chown -R agent: .
cp infra/systemd/moex-agent.service infra/systemd/moex-bot.service \
   infra/systemd/moex-agent-backup.service infra/systemd/moex-agent-backup.timer /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now moex-agent.service moex-bot.service moex-agent-backup.timer
```

## 4. First-run smoke checklist

Run one cycle by hand, then confirm both services are healthy and the bot answers. (Docker: prefix
with `docker compose -f infra/docker-compose.yml exec agent`; systemd: `sudo -u agent .venv/bin/python`.)

- [ ] **Manual cycle runs:** `python -m agent.src.cli run-eod --force` → status `completed` (or
      `halted` if data isn't seeded yet — that's the gate working, not a failure).
- [ ] **State persisted:** `python -m agent.src.cli status` shows `last_successful_cycle`, the
      live/shadow position tracks, and `recent_shadow_orders` (paper-shadow activity, not emptiness).
- [ ] **Integrity gate present:** `status` / the EOD digest reports integrity OK or a HALT with reasons.
- [ ] **Paper lock verified:** `status` shows `mode=paper`, `live_enabled=false`.
- [ ] **Bot is the only poller:** `docker compose ... logs bot` shows it long-polling, no
      `Conflict: terminated by other getUpdates` error (would mean a second poller on the token).
- [ ] **Bot answers** from an admin chat: **`/status`** (mode, kill-switch, live/shadow gross) and
      **`/shadowlog`** (recent forward-shadow cycles). Also try `/gate`, `/pnl`, `/integrity`.
- [ ] **Scheduler armed:** agent logs show `scheduler up: tz=Europe/Moscow eod=19:05 preopen=09:30`.
- [ ] **Restart recovers:** `docker compose ... restart agent` (or `systemctl restart`) → `status`
      still shows the prior book + cycle (state is durable).

The scheduler then fires the EOD cycle automatically each trading day (RU-holiday-aware) and the
forward-shadow track accrues in `data/agent/shadow_pnl.jsonl`.

## 5. Real forward-shadow accrual (flip blocks live — STILL paper)

Out of the box the blocks are mocked (the cycle runs end-to-end with zero seeded data — good for
bring-up + smoke). To accrue the **real** H9 forward-shadow track off real prices + the dividend
feed, flip the upstream blocks to live via `.env` (no image rebuild — `agent/src/config.py` reads
these). This is still **paper**: the shadow gate keeps H9 at zero live capital and execution stays a
paper sim.

```ini
AGENT_BACKEND_MODE=live          # real MOEX ingest + integrity gate (needs data/raw seeded)
AGENT_SLEEVE_MODE=live           # real H9 sleeve via ml/scripts/predict_dividend_sleeve.py
AGENT_COMBINER_MODE=live         # real risk_manager combiner + H4/H5 risk analytics
# execution stays the paper sim (agent_config.json blocks.execution.mode=live, broker sim)
AGENT_LLM_REFRESH_CMD=python llm/scripts/refresh_dividend_feed.py   # EOD step 2 feed refresh
```

One-time data seed (otherwise the integrity gate HALTs — correct, but no signal accrues):

```bash
docker compose -f infra/docker-compose.yml exec agent python -m backend.ingest --backfill
# then the daily EOD ingest keeps it fresh
```

## Monitoring, backups, logs

- **Alerts (push):** `AGENT_ALERT_CHANNEL=telegram` sends the EOD digest, data-HALT, and
  dead-man's-switch to `TELEGRAM_CHAT_ID`. `stdout` (default) needs no secrets.
- **Bot (pull):** read-only commands `/status /positions /pnl /prices /gate /shadowlog /cycle
  /integrity`; admin-only `/users /allow /deny` manage the read allowlist at runtime.
- **Health:** compose healthchecks (`agent status`, bot import) + `restart: unless-stopped`.
- **Backups:** `infra/backup.sh` → daily systemd timer (Option B), or host cron under Docker:
  `0 20 * * * docker compose -f /opt/moex-candle-predictor/infra/docker-compose.yml exec -T agent bash infra/backup.sh`.
- **Logs:** `data/agent/logs/agent.log` (rotating 10×5 MB) + `docker compose ... logs`.

## Secrets discipline

Only in `/etc/moex-agent.env` (systemd `EnvironmentFile`, `chmod 600`) or the repo-root `.env`
(Docker `env_file`). Both are gitignored. Never commit a real token.

---

## ⛔ LIVE TRADING — DO NOT ENABLE (separate, gated)

**Live is OFF in this deploy and must stay off until sign-off.** The paper lock is two independent
flags, both default-off; leave them off:

```ini
AGENT_ENABLE_LIVE=false      # agent-side gate (agent/src/config.py)
EXECUTION_ALLOW_LIVE=0       # execution-side gate (execution/src/config.py)
```

`is_production=false` flows through every artifact regardless. Enabling live requires ALL of, in
order: (1) backend autonomous ≥2 weeks clean; (2) a full dividend season paper-run on the VDS;
(3) the **forward-shadow gate MET** — realized run-up net>0 consistent with history (the
`/shadowlog` track + `data/reports/h9_shadow_pnl.txt`); (4) verified broker FIGIs + sandbox
wire-test; (5) **team sign-off**. Only then, as a deliberate separate change: set
`AGENT_MODE=live` **and** `AGENT_ENABLE_LIVE=true` **and** `EXECUTION_ALLOW_LIVE=1` with a
full-access `TINVEST_TOKEN` + `TINVEST_ACCOUNT_ID`. Any one flag missing → stays paper. Source of
truth: `docs/VDS_AUTONOMOUS_PLAN.md`.
