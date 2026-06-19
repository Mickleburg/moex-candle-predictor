# Deploy — MOEX autonomous agent on a VDS

How to bring the V3 daily cycle up on a Linux VDS (Ubuntu LTS) and keep it alive. The agent
itself is `agent/` (orchestrator + scheduler); this dir is the runtime around it.

> Decisions made here (the umbrella plan delegated them): **Docker Compose** as the primary
> one-command deploy + supervisor, **systemd** as the venv alternative; **in-process
> APScheduler** for the EOD/pre-open triggers + dead-man's-switch (lives next to the state
> store and the trading-day gate); **SQLite** state store; **Telegram** alerts. Live trading
> stays disabled until shadow-gate + sign-off (`AGENT_ENABLE_LIVE`).

## Option A — Docker Compose (recommended)

```bash
git clone <repo> && cd moex-candle-predictor
cp .env.example .env            # fill in alerts/broker secrets
docker compose -f infra/docker-compose.yml up -d --build
```

That's the whole bring-up. `restart: unless-stopped` is the supervisor; `./data` is a
bind-mount so the SQLite store, cycle results, logs and parquet candles survive rebuilds and
reboots (state recovers on restart). Useful commands:

```bash
docker compose -f infra/docker-compose.yml logs -f agent          # tail logs
docker compose -f infra/docker-compose.yml exec agent python -m agent.src.cli status
docker compose -f infra/docker-compose.yml exec agent python -m agent.src.cli run-eod --force
docker compose -f infra/docker-compose.yml exec agent python -m agent.src.cli kill-switch on
```

## Option B — venv + systemd

```bash
sudo git clone <repo> /opt/moex-candle-predictor && cd /opt/moex-candle-predictor
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
sudo useradd --system --home /opt/moex-candle-predictor agent && sudo chown -R agent: .
sudo cp .env.example /etc/moex-agent.env && sudo chmod 600 /etc/moex-agent.env   # fill in
sudo cp infra/systemd/moex-agent.service /etc/systemd/system/
sudo cp infra/systemd/moex-agent-backup.* /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now moex-agent.service moex-agent-backup.timer
```

## Scheduler

In-process APScheduler (`agent/src/scheduler.py`), TZ `Europe/Moscow`, gated to MOEX trading
days (RU holidays via the shared backend calendar):

| Trigger | Default | Action |
|---------|---------|--------|
| EOD | 19:05 Mon–Fri | `run_eod_cycle` — ingest → integrity → sleeve → combine → execute(paper) → persist → digest |
| pre-open | 09:30 Mon–Fri | `run_preopen` — kill-switch, overnight-gap/HALT check, confirm/cancel limit orders |
| dead-man's-switch | every 30 min | alert if the most recent **due** EOD cycle never completed |

Times/timezone are in `agent/config/agent_config.json` (`schedule`). Prefer cron/systemd-timers?
The same triggers map to `run-eod` / `run-preopen` CLI calls — but you then lose the unified
in-process dead-man's-switch, so the daemon is recommended.

## Monitoring & alerts

`AGENT_ALERT_CHANNEL=telegram` + `TELEGRAM_BOT_TOKEN`/`TELEGRAM_CHAT_ID` sends the EOD digest
(entries/exits, gate status, per-sleeve P&L, data failures), data-HALT alerts, and the
dead-man's-switch alert. Default `stdout` needs no secrets (dev/paper). Health: the compose
`healthcheck` runs `agent status`; logs rotate under `data/agent/logs/` (10×5 MB).

## Backups

`infra/backup.sh` takes a consistent SQLite hot-backup of the state store + tars cycle results
+ copies the shadow log into `data/agent/backups/<UTC>/`, keeping the newest `BACKUP_KEEP`
(14). Wired to a daily systemd timer (Option B); under Docker run it via cron on the host:
`0 20 * * * docker compose -f /path/infra/docker-compose.yml exec -T agent bash infra/backup.sh`.

## Secrets

Only ever in `.env` (Docker `env_file`) or `/etc/moex-agent.env` (systemd `EnvironmentFile`,
`chmod 600`). Both are gitignored. Never commit real tokens. See `.env.example`.

## Going live (gated)

`is_production=false` flows through every artifact until: backend autonomous ≥2 weeks clean ·
paper-cycle a dividend season on the VDS · forward-shadow gate net>0 consistent with history ·
team sign-off. Only then set `AGENT_ENABLE_LIVE=true` **and** `AGENT_MODE=live` (both required;
either alone stays paper) and wire `blocks.execution` to the real broker adapter. Source of
truth: `docs/VDS_AUTONOMOUS_PLAN.md`.
