# Forward-accrual snapshot: VDS → local (leak-safe) + feed-refresh smoke

How to pull the forward evidence the paper agent accumulates on the VDS (the shadow track, state,
reports, the dividend feed) to a local machine for inspection — **without touching the production
store and without it becoming a tuning input** — and how to confirm the dividend feed actually
refreshes inside the daily cycle (so July records reach the feed and the shadow track accrues).

> **Read-only, measurement only.** A forward snapshot is evidence for the shadow gate — never an
> input to tune/select H9. Optimising on the forward period burns the forward gate. Both scripts
> stamp/keep this: the manifest carries a `not_for_tuning` banner and the landed copy is chmod
> read-only and self-ignored (never committed).

## 1. Export on the VDS (read-only)

```bash
# inside the agent container, or the venv checkout
python scripts/export_forward_snapshot.py                 # -> data/forward_exports/<UTCdate>/
# docker:
docker compose -f infra/docker-compose.yml exec agent python scripts/export_forward_snapshot.py
```

Captures (reads only; `state.sqlite` via a consistent online backup opened `mode=ro`):

| In snapshot | Source |
|-------------|--------|
| `state.sqlite` | agent state store (positions, P&L, cycle_runs) — consistent backup |
| `shadow_pnl.jsonl` | the forward-shadow track |
| `reports/*.json`, `reports/h9_shadow_pnl.txt` | integrity/ingest reports + realised shadow-gate verdict |
| `feed/dividend_calendar_upcoming.csv`, `feed/dividend_announcements.csv`, `feed/dividends.csv` | forward feed + realised dividend events |
| `feed/edisclosure/*.parquet` | raw disclosure titles — so the no-lookahead verifier can re-check the feed locally |

It writes **only** under `--out` (never `data/agent` or `data/raw` — it refuses), is idempotent
(same data → same SHA256s), and prints the `manifest.json` to stdout. The manifest records
`created_at`, `git_commit`, host, the shadow-track `as_of_range`, a SHA256 per file, and the
`not_for_tuning` banner.

## 2. Transfer

```bash
scp -r <vds>:/opt/moex-candle-predictor/data/forward_exports/<date> ./incoming_snapshot
# (docker: docker compose cp moex-agent:/app/data/forward_exports/<date> ./incoming_snapshot)
```

## 3. Import + verify locally (fails on any discrepancy)

```bash
python scripts/import_forward_snapshot.py ./incoming_snapshot      # -> data/forward_snapshots/<date>/
```

Lands under `data/forward_snapshots/<date>/` as **read-only** only if ALL pass — else exits non-zero:

1. **SHA256** — every manifest file present + matches (integrity / truncated-transfer / tamper).
2. **Shadow-track no-lookahead + monotonicity** — `as_of` non-decreasing (no rewind), each
   `trade_date ≤ as_of`, no duplicate `trade_date`.
3. **Dividend-feed no-lookahead** — REUSES `llm/scripts/verify_dividend_feed` against the bundled
   raw disclosure parquet (`board_reco_date ≤ as_of`, etc.). Skipped with a warning only if its
   inputs/deps are absent — never silently passed.

`data/forward_snapshots/` self-ignores (a `.gitignore` of `*`), so evidence is never committed.

## 4. Feed auto-refresh in the daily cycle (Part C)

The shadow track only accrues new ex-dates if EOD **step 2** refreshes the dividend feed. The
orchestrator runs `blocks.llm.refresh_cmd` before the sleeve (best-effort; a failure alerts but
never blocks trading), and `agent/src/config.py` overlays `AGENT_LLM_REFRESH_CMD` from the env so a
deploy wires it without editing the baked config.

**On the VDS, set it** in `/etc/moex-agent.env` (or the Docker `.env`):

```ini
AGENT_LLM_REFRESH_CMD=python llm/scripts/refresh_dividend_feed.py
# the orchestrator appends --as-of <trade_date> automatically (blocks.llm.pass_as_of=true in config)
```

Without it the feed never updates → July board recommendations never reach
`dividend_calendar_upcoming.csv` → the sleeve sees no new ex-dates → **no accrual**.

### Smoke: "the feed refreshed during the cycle"

The cycle records the refresh outcome in `agent_cycle_result.risk_summary.feed_refresh`
(`{configured, ran, ok, changed, upcoming, rc}`). After a cycle:

```bash
docker compose -f infra/docker-compose.yml exec agent python -m agent.src.cli run-eod --force
# look at risk_summary.feed_refresh in the printed result, or:
docker compose -f infra/docker-compose.yml exec agent python -m agent.src.cli status   # last_cycle
```

- [ ] `feed_refresh.configured == true` (AGENT_LLM_REFRESH_CMD is set).
- [ ] `feed_refresh.ran == true` and `rc == 0` (the refresh executed cleanly).
- [ ] `feed_refresh.changed` is `true` on a day with new disclosures (or `false` = up-to-date,
      `upcoming` = current feed row count) — and the EOD digest line `feed_refresh=…` shows it.
- [ ] the bot `/shadowlog` shows the cycle accruing; `/gate` reflects the latest forward P&L.

If `feed_refresh.configured == false`, the env var is missing on the VDS — fix it, or the season
accrues nothing. (Add this row to the deploy smoke checklist in `infra/README.md`.)
