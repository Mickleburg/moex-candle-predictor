"""Durable agent state — SQLite (stdlib, transactional, survives restart).

Chosen over flat files because the agent must be idempotent and recover after a crash:
a cycle is keyed by (trade_date, phase) with a UNIQUE constraint, so re-running the same
day is a safe no-op; the current book, open orders, fills and per-SLEEVE P&L attribution
are all queryable. Everything is one file (data/agent/state.sqlite) — easy to back up.

Tables:
  cycle_runs        one row per (trade_date, phase); status + the persisted cycle result
  positions         current paper/live book (ticker -> lots, avg price, sleeve attribution)
  orders            every order intent + status (client_order_id is the dedup key)
  executions        execution_report rows returned by the execution block
  pnl_attribution   realized/unrealized P&L per sleeve per day (sleeve with no edge -> 0)
  kv                small key/value flags (kill_switch, last heartbeat, ...)
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cycle_runs (
    trade_date   TEXT NOT NULL,
    phase        TEXT NOT NULL,
    status       TEXT NOT NULL,           -- running | completed | halted | killed | failed
    mode         TEXT,
    block_mode   TEXT,
    as_of        TEXT,
    halt_reason  TEXT,
    result_json  TEXT,
    started_at   TEXT NOT NULL,
    finished_at  TEXT,
    PRIMARY KEY (trade_date, phase)
);
CREATE TABLE IF NOT EXISTS positions (
    ticker                   TEXT NOT NULL,
    capital_state            TEXT NOT NULL DEFAULT 'live',   -- 'live' (passed gate) | 'shadow' (gated out)
    lots                     INTEGER NOT NULL DEFAULT 0,
    avg_price                REAL NOT NULL DEFAULT 0,
    last_price               REAL,
    is_hedge                 INTEGER NOT NULL DEFAULT 0,
    sleeve_contributions     TEXT,                       -- JSON {sleeve: weight}
    updated_at               TEXT NOT NULL,
    PRIMARY KEY (ticker, capital_state)
);
CREATE TABLE IF NOT EXISTS orders (
    client_order_id  TEXT PRIMARY KEY,
    trade_date       TEXT NOT NULL,
    phase            TEXT,
    ticker           TEXT NOT NULL,
    side             TEXT NOT NULL,
    quantity_lots    INTEGER NOT NULL,
    order_type       TEXT NOT NULL,
    limit_price      REAL,
    status           TEXT NOT NULL,
    created_at       TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS executions (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    client_order_id       TEXT NOT NULL,
    ticker                TEXT NOT NULL,
    status                TEXT NOT NULL,
    filled_quantity_lots  INTEGER NOT NULL DEFAULT 0,
    avg_fill_price        REAL,
    message               TEXT,
    ts                    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS pnl_attribution (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_date      TEXT NOT NULL,
    sleeve          TEXT NOT NULL,
    capital_state   TEXT NOT NULL DEFAULT 'live',   -- 'live' vs 'shadow' capital P&L (kept separate)
    realized_pnl    REAL NOT NULL DEFAULT 0,
    unrealized_pnl  REAL NOT NULL DEFAULT 0,
    gross           REAL NOT NULL DEFAULT 0,
    ts              TEXT NOT NULL,
    UNIQUE (trade_date, sleeve, capital_state)
);
CREATE TABLE IF NOT EXISTS kv (
    key         TEXT PRIMARY KEY,
    value       TEXT,
    updated_at  TEXT NOT NULL
);
"""


# Migrations for DBs predating the capital_state (shadow-gate) split — preserve rows as 'live'.
_REBUILD_POSITIONS = """
ALTER TABLE positions RENAME TO positions_old;
CREATE TABLE positions (
    ticker TEXT NOT NULL, capital_state TEXT NOT NULL DEFAULT 'live', lots INTEGER NOT NULL DEFAULT 0,
    avg_price REAL NOT NULL DEFAULT 0, last_price REAL, is_hedge INTEGER NOT NULL DEFAULT 0,
    sleeve_contributions TEXT, updated_at TEXT NOT NULL, PRIMARY KEY (ticker, capital_state)
);
INSERT INTO positions (ticker, capital_state, lots, avg_price, last_price, is_hedge, sleeve_contributions, updated_at)
    SELECT ticker, 'live', lots, avg_price, last_price, is_hedge, sleeve_contributions, updated_at FROM positions_old;
DROP TABLE positions_old;
"""
_REBUILD_PNL = """
ALTER TABLE pnl_attribution RENAME TO pnl_attribution_old;
CREATE TABLE pnl_attribution (
    id INTEGER PRIMARY KEY AUTOINCREMENT, trade_date TEXT NOT NULL, sleeve TEXT NOT NULL,
    capital_state TEXT NOT NULL DEFAULT 'live', realized_pnl REAL NOT NULL DEFAULT 0,
    unrealized_pnl REAL NOT NULL DEFAULT 0, gross REAL NOT NULL DEFAULT 0, ts TEXT NOT NULL,
    UNIQUE (trade_date, sleeve, capital_state)
);
INSERT INTO pnl_attribution (trade_date, sleeve, capital_state, realized_pnl, unrealized_pnl, gross, ts)
    SELECT trade_date, sleeve, 'live', realized_pnl, unrealized_pnl, gross, ts FROM pnl_attribution_old;
DROP TABLE pnl_attribution_old;
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class StateStore:
    """Thread-safe SQLite wrapper for the agent's durable state."""

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.executescript(_SCHEMA)
        self._migrate()
        self._conn.commit()

    def _migrate(self) -> None:
        """Add the capital_state dimension to DBs created before the shadow-gate split.

        Old `positions` / `pnl_attribution` rows are rebuilt as capital_state='live' (they
        predate the gate, so they were live-intent). The state store is regenerable, but we
        preserve paper positions across the upgrade rather than dropping them.
        """
        for table, rebuild in (("positions", _REBUILD_POSITIONS), ("pnl_attribution", _REBUILD_PNL)):
            cols = [r[1] for r in self._conn.execute(f"PRAGMA table_info({table})").fetchall()]
            if cols and "capital_state" not in cols:
                self._conn.executescript(rebuild)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            try:
                yield self._conn
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    # --- cycle lifecycle (idempotency + recovery) --------------------------------------
    def begin_cycle(self, trade_date: str, phase: str, *, mode: str, block_mode: str,
                    as_of: str, force: bool = False) -> dict[str, Any]:
        """Claim a (trade_date, phase) cycle.

        Returns {"started": bool, "prior": row|None}. started=False means a completed/halted
        run already exists -> the caller should skip (idempotent), unless force=True, which
        re-opens the slot for a fresh run.
        """
        with self._tx() as c:
            row = c.execute(
                "SELECT * FROM cycle_runs WHERE trade_date=? AND phase=?",
                (trade_date, phase),
            ).fetchone()
            if row is not None and not force:
                if row["status"] in {"completed", "halted", "killed"}:
                    return {"started": False, "prior": dict(row)}
                # a stale 'running'/'failed' row -> reclaim it (crash recovery).
            c.execute(
                "INSERT INTO cycle_runs (trade_date, phase, status, mode, block_mode, as_of, started_at) "
                "VALUES (?,?,?,?,?,?,?) "
                "ON CONFLICT(trade_date, phase) DO UPDATE SET "
                "status=excluded.status, mode=excluded.mode, block_mode=excluded.block_mode, "
                "as_of=excluded.as_of, started_at=excluded.started_at, finished_at=NULL, "
                "halt_reason=NULL, result_json=NULL",
                (trade_date, phase, "running", mode, block_mode, as_of, _now()),
            )
            return {"started": True, "prior": dict(row) if row else None}

    def finish_cycle(self, trade_date: str, phase: str, status: str,
                     *, result: Optional[dict] = None, halt_reason: Optional[str] = None) -> None:
        with self._tx() as c:
            c.execute(
                "UPDATE cycle_runs SET status=?, finished_at=?, halt_reason=?, result_json=? "
                "WHERE trade_date=? AND phase=?",
                (status, _now(), halt_reason,
                 json.dumps(result, ensure_ascii=False) if result is not None else None,
                 trade_date, phase),
            )

    def get_cycle(self, trade_date: str, phase: str) -> Optional[dict]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM cycle_runs WHERE trade_date=? AND phase=?", (trade_date, phase)
            ).fetchone()
            return dict(row) if row else None

    def last_successful_cycle(self) -> Optional[dict]:
        """Most recent completed/halted cycle (dead-man's-switch reference)."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM cycle_runs WHERE status IN ('completed','halted') "
                "ORDER BY finished_at DESC LIMIT 1"
            ).fetchone()
            return dict(row) if row else None

    # --- positions ---------------------------------------------------------------------
    def get_positions(self, capital_state: str | None = "live") -> list[dict]:
        """Positions for a capital track ('live' default | 'shadow'); None = both tracks."""
        with self._lock:
            if capital_state is None:
                return [dict(r) for r in self._conn.execute("SELECT * FROM positions").fetchall()]
            return [dict(r) for r in self._conn.execute(
                "SELECT * FROM positions WHERE capital_state=?", (capital_state,)).fetchall()]

    def upsert_position(self, ticker: str, lots: int, avg_price: float, last_price: float | None,
                        *, is_hedge: bool = False, sleeve_contributions: dict | None = None,
                        capital_state: str = "live") -> None:
        with self._tx() as c:
            if lots == 0:
                c.execute("DELETE FROM positions WHERE ticker=? AND capital_state=?",
                          (ticker, capital_state))
                return
            c.execute(
                "INSERT INTO positions (ticker, capital_state, lots, avg_price, last_price, is_hedge, "
                "sleeve_contributions, updated_at) VALUES (?,?,?,?,?,?,?,?) "
                "ON CONFLICT(ticker, capital_state) DO UPDATE SET lots=excluded.lots, "
                "avg_price=excluded.avg_price, last_price=excluded.last_price, is_hedge=excluded.is_hedge, "
                "sleeve_contributions=excluded.sleeve_contributions, updated_at=excluded.updated_at",
                (ticker, capital_state, int(lots), float(avg_price), last_price, 1 if is_hedge else 0,
                 json.dumps(sleeve_contributions or {}, ensure_ascii=False), _now()),
            )

    # --- orders (dedup by client_order_id) ---------------------------------------------
    def order_exists(self, client_order_id: str) -> bool:
        with self._lock:
            return self._conn.execute(
                "SELECT 1 FROM orders WHERE client_order_id=?", (client_order_id,)
            ).fetchone() is not None

    def record_order(self, order: dict, *, trade_date: str, phase: str, status: str) -> None:
        with self._tx() as c:
            c.execute(
                "INSERT INTO orders (client_order_id, trade_date, phase, ticker, side, "
                "quantity_lots, order_type, limit_price, status, created_at) VALUES (?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(client_order_id) DO UPDATE SET status=excluded.status",
                (order["client_order_id"], trade_date, phase, order["ticker"], order["side"],
                 int(order["quantity_lots"]), order["order_type"], order.get("limit_price"),
                 status, _now()),
            )

    def open_orders(self) -> list[dict]:
        """Resting orders awaiting a fill/cancel (PLACED). DRY_RUN never reaches the book."""
        with self._lock:
            return [dict(r) for r in self._conn.execute(
                "SELECT * FROM orders WHERE status IN ('PLACED','DRY_RUN')").fetchall()]

    def all_orders(self) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self._conn.execute("SELECT * FROM orders").fetchall()]

    def set_order_status(self, client_order_id: str, status: str) -> None:
        with self._tx() as c:
            c.execute("UPDATE orders SET status=? WHERE client_order_id=?", (status, client_order_id))

    # --- executions --------------------------------------------------------------------
    def record_execution(self, report: dict) -> None:
        with self._tx() as c:
            c.execute(
                "INSERT INTO executions (client_order_id, ticker, status, filled_quantity_lots, "
                "avg_fill_price, message, ts) VALUES (?,?,?,?,?,?,?)",
                (report["client_order_id"], report["ticker"], report["status"],
                 int(report.get("filled_quantity_lots", 0)), report.get("avg_fill_price"),
                 report.get("message", ""), _now()),
            )

    # --- per-sleeve P&L attribution (live vs shadow capital kept separate) --------------
    def record_pnl_attribution(self, trade_date: str, sleeve: str, *, realized: float,
                               unrealized: float, gross: float, capital_state: str = "live") -> None:
        with self._tx() as c:
            c.execute(
                "INSERT INTO pnl_attribution (trade_date, sleeve, capital_state, realized_pnl, "
                "unrealized_pnl, gross, ts) VALUES (?,?,?,?,?,?,?) "
                "ON CONFLICT(trade_date, sleeve, capital_state) DO UPDATE SET "
                "realized_pnl=excluded.realized_pnl, unrealized_pnl=excluded.unrealized_pnl, "
                "gross=excluded.gross, ts=excluded.ts",
                (trade_date, sleeve, capital_state, float(realized), float(unrealized),
                 float(gross), _now()),
            )

    def pnl_by_sleeve(self, capital_state: str | None = None) -> list[dict]:
        """Cumulative P&L per (sleeve, capital_state); pass capital_state to filter one track."""
        with self._lock:
            if capital_state is None:
                return [dict(r) for r in self._conn.execute(
                    "SELECT sleeve, capital_state, SUM(realized_pnl) AS realized, "
                    "SUM(unrealized_pnl) AS unrealized FROM pnl_attribution "
                    "GROUP BY sleeve, capital_state ORDER BY sleeve, capital_state").fetchall()]
            return [dict(r) for r in self._conn.execute(
                "SELECT sleeve, capital_state, SUM(realized_pnl) AS realized, "
                "SUM(unrealized_pnl) AS unrealized FROM pnl_attribution WHERE capital_state=? "
                "GROUP BY sleeve ORDER BY sleeve", (capital_state,)).fetchall()]

    def forward_pnl_by_sleeve(self, capital_state: str = "live") -> dict[str, dict]:
        """Per-sleeve forward P&L for the combiner's shadow gate seam: {sleeve: {"forward_pnl": x}}.

        Defaults to the LIVE track — a production sleeve whose live forward P&L turns negative is
        demoted back to shadow (invariant #9). Pass 'shadow' to inspect the shadow accrual.
        """
        out: dict[str, dict] = {}
        for row in self.pnl_by_sleeve(capital_state):
            out[row["sleeve"]] = {"forward_pnl": (row["realized"] or 0.0) + (row["unrealized"] or 0.0)}
        return out

    # --- key/value flags (kill-switch, heartbeat) --------------------------------------
    def set_flag(self, key: str, value: Any) -> None:
        with self._tx() as c:
            c.execute(
                "INSERT INTO kv (key, value, updated_at) VALUES (?,?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                (key, json.dumps(value), _now()),
            )

    def get_flag(self, key: str, default: Any = None) -> Any:
        with self._lock:
            row = self._conn.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
            return json.loads(row["value"]) if row and row["value"] is not None else default

    # --- kill-switch convenience -------------------------------------------------------
    def kill_switch_engaged(self) -> bool:
        return bool(self.get_flag("kill_switch", False))

    def set_kill_switch(self, engaged: bool) -> None:
        self.set_flag("kill_switch", bool(engaged))
