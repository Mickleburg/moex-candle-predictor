"""Read-only views over the agent's state + regenerable reports.

The agent OWNS its SQLite state store (agent/src/state_store.py); this bot only reads it, so it
opens the DB with a ``mode=ro`` URI — it never creates tables, migrates, or writes. The SELECTs
here mirror the schema and the read methods of ``StateStore`` (the source of truth for the
shape); if that schema changes, update both. Every accessor degrades gracefully: a missing DB or
report returns an empty/None result instead of raising, so a command can render "no data yet".

Data surfaces (per the bot task):
  * state.sqlite          positions (live|shadow), open orders, per-sleeve P&L, kv flags, cycles
  * data_integrity_status.json   OK/HALT data gate
  * h9_shadow_pnl.txt      realized-P&L shadow-gate verdict (is_production / MET|NOT_MET)
  * data/raw/*.parquet     last close per ticker (via backend.store, no network)
"""

from __future__ import annotations

import json
import re
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

from .config import BotConfig


class ReadOnlyState:
    """Read-only accessor for the agent's SQLite state store (opened with mode=ro)."""

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)

    def available(self) -> bool:
        return self.db_path.exists()

    @contextmanager
    def _conn(self) -> Iterator[Optional[sqlite3.Connection]]:
        if not self.db_path.exists():
            yield None
            return
        uri = f"file:{self.db_path.as_posix()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _query(self, sql: str, params: tuple = ()) -> list[dict]:
        try:
            with self._conn() as conn:
                if conn is None:
                    return []
                return [dict(r) for r in conn.execute(sql, params).fetchall()]
        except sqlite3.Error:
            # corrupt / locked / pre-schema DB — treat as "no data" rather than crashing a command.
            return []

    # --- key/value flags (kill-switch, last_cycle heartbeat) ---------------------------
    def get_flag(self, key: str, default: Any = None) -> Any:
        rows = self._query("SELECT value FROM kv WHERE key=?", (key,))
        if not rows or rows[0]["value"] is None:
            return default
        try:
            return json.loads(rows[0]["value"])
        except (json.JSONDecodeError, TypeError):
            return default

    def kill_switch_engaged(self) -> bool:
        return bool(self.get_flag("kill_switch", False))

    def last_cycle(self) -> Optional[dict]:
        return self.get_flag("last_cycle")

    def last_successful_cycle(self) -> Optional[dict]:
        rows = self._query(
            "SELECT * FROM cycle_runs WHERE status IN ('completed','halted') "
            "ORDER BY finished_at DESC LIMIT 1"
        )
        return rows[0] if rows else None

    def latest_cycle(self, phase: str = "eod") -> Optional[dict]:
        """Most recent run of a phase, with its persisted result_json parsed into ``result``."""
        rows = self._query(
            "SELECT * FROM cycle_runs WHERE phase=? ORDER BY started_at DESC LIMIT 1", (phase,)
        )
        if not rows:
            return None
        row = rows[0]
        rj = row.get("result_json")
        row["result"] = json.loads(rj) if rj else None
        return row

    # --- positions (live default | shadow | both) --------------------------------------
    def positions(self, capital_state: str | None = "live") -> list[dict]:
        if capital_state is None:
            return self._query("SELECT * FROM positions ORDER BY ticker")
        return self._query(
            "SELECT * FROM positions WHERE capital_state=? ORDER BY ticker", (capital_state,)
        )

    def gross(self, capital_state: str) -> float:
        """Book gross (sum |lots * last_price|) for a capital track, marked at last_price."""
        split = self.gross_split(capital_state)
        return split["total"]

    def gross_split(self, capital_state: str) -> dict[str, float]:
        """Marked gross split into directional vs hedge legs (so the two are never conflated).

        Returns {"directional", "hedge", "total"}. The directional gross is the real economic
        exposure; the hedge legs offset it, so a combined gross can read >100% of capital and is
        misleading on its own — /status shows them separately.
        """
        directional = hedge = 0.0
        for p in self.positions(capital_state):
            last = p.get("last_price")
            if last is None:
                last = p.get("avg_price") or 0.0
            notional = abs(int(p.get("lots", 0)) * float(last))
            if p.get("is_hedge"):
                hedge += notional
            else:
                directional += notional
        return {"directional": directional, "hedge": hedge, "total": directional + hedge}

    # --- per-sleeve P&L (live vs shadow kept separate) ---------------------------------
    def pnl_by_sleeve(self, capital_state: str | None = None) -> list[dict]:
        """Cumulative realized+unrealized P&L per (sleeve, capital_state)."""
        if capital_state is None:
            return self._query(
                "SELECT sleeve, capital_state, SUM(realized_pnl) AS realized, "
                "SUM(unrealized_pnl) AS unrealized FROM pnl_attribution "
                "GROUP BY sleeve, capital_state ORDER BY sleeve, capital_state"
            )
        return self._query(
            "SELECT sleeve, capital_state, SUM(realized_pnl) AS realized, "
            "SUM(unrealized_pnl) AS unrealized FROM pnl_attribution WHERE capital_state=? "
            "GROUP BY sleeve ORDER BY sleeve", (capital_state,)
        )

    def open_orders(self) -> list[dict]:
        return self._query(
            "SELECT * FROM orders WHERE status IN ('PLACED','DRY_RUN') ORDER BY created_at"
        )


def read_integrity(path: Path | str) -> Optional[dict]:
    """Parse the data-integrity report (OK/HALT + reasons). None if absent/unreadable."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


_VERDICT_RE = re.compile(r"VERDICT:\s*(MET|NOT MET)", re.IGNORECASE)
_FORWARD_RE = re.compile(r"FORWARD\s*:\s*n=\s*(\d+)\s+net\s+([+\-]?[0-9.]+)\s+%pos\s+([0-9.]+)")
_PROD_RE = re.compile(r"is_production\s*=?\s*(false|true)", re.IGNORECASE)


def read_gate(path: Path | str) -> dict:
    """Parse the shadow-gate verdict report into {is_production, met, forward_*}.

    The report is the ML chat's h9_shadow_pnl text artifact. We extract just the headline gate
    state; ``found=False`` means the report is missing (the command then falls back to the DB
    shadow track and the standing invariant is_production=false).
    """
    p = Path(path)
    if not p.exists():
        return {"found": False, "is_production": False}
    text = p.read_text(encoding="utf-8")
    out: dict[str, Any] = {"found": True, "is_production": False}

    m = _PROD_RE.search(text)
    if m:
        out["is_production"] = m.group(1).lower() == "true"
    m = _VERDICT_RE.search(text)
    if m:
        out["met"] = m.group(1).upper() == "MET"
    m = _FORWARD_RE.search(text)
    if m:
        out["forward_n"] = int(m.group(1))
        out["forward_net"] = float(m.group(2))
        out["forward_pct_pos"] = float(m.group(3))
    return out


def read_shadow_log(path: Path | str, limit: int = 5) -> list[dict]:
    """Last ``limit`` records of the forward-shadow track (data/agent/shadow_pnl.jsonl).

    The orchestrator appends one no-lookahead line per cycle (agent/src/pnl.append_shadow_log):
    {trade_date, as_of, sleeves, regime, book, sleeve_pnl{sleeve:{unrealized,gross}}}. This is the
    accrual the shadow gate watches in the July dividend season. Returns newest-last; malformed
    lines are skipped, a missing file returns [].
    """
    p = Path(path)
    if not p.exists():
        return []
    try:
        lines = p.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    records: list[dict] = []
    for line in lines[-max(limit, 0):]:
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def last_close(ticker: str, timeframe: str, data_raw: Path | str) -> Optional[float]:
    """Last stored close for a ticker (no network — reads the backend candle store)."""
    try:
        from backend import store
    except Exception:  # noqa: BLE001 - backend/pandas optional for a price-less deploy
        return None
    try:
        df = store.load_ticker(ticker, timeframe, data_dir=Path(data_raw))
    except Exception:  # noqa: BLE001 - a malformed parquet must not crash /prices
        return None
    if df is None or df.empty:
        return None
    return float(df["close"].iloc[-1])


def sector_of(ticker: str) -> str:
    """Sector index for a ticker (mirrors the combiner's hedge mapping). 'n/a' if unavailable."""
    try:
        from risk_manager.src.sectors import sector_of as _sector_of
    except Exception:  # noqa: BLE001
        return "n/a"
    return _sector_of(ticker)


def make_state(config: BotConfig) -> ReadOnlyState:
    return ReadOnlyState(config.state_db)
