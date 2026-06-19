"""Live block adapters — call the real blocks (read-only) and validate their JSON.

- LiveSleeve / LiveCombiner import the ml + risk_manager PUBLIC functions (the same handshake
  risk_manager/scripts/demo_combine_h9.py uses) — read-only, no edits to their code. Imports
  are lazy so the orchestrator core stays stdlib-only until the live path is actually used.
- LiveBackend / LiveExecution shell out to the backend/execution block CLIs (parallel chats)
  via configured commands and parse the contract JSON they emit. Until those CLIs exist, the
  registry keeps these blocks on the mock path; LiveExecution refuses to silently no-op.

Everything below is INFORMATION flowing on contracts; is_production stays false until sign-off.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

from .. import contracts
from .base import ExecutionResult, IntegrityStatus

REPO_ROOT = Path(__file__).resolve().parents[3]


def _run_json_cmd(command: list[str], payload: Optional[dict] = None, timeout: float = 600.0) -> dict:
    """Run a block CLI, feed it JSON on stdin (optional), parse JSON from stdout."""
    proc = subprocess.run(
        command, input=json.dumps(payload) if payload is not None else None,
        capture_output=True, text=True, timeout=timeout, cwd=str(REPO_ROOT),
    )
    if proc.returncode != 0:
        raise RuntimeError(f"command {command!r} failed (rc={proc.returncode}): {proc.stderr.strip()}")
    return json.loads(proc.stdout)


class LiveSleeve:
    """The real ML H9 sleeve: load_panels -> build_sleeve_signal(as_of)."""

    def __init__(self, universe: list[str], timeframe: str = "1D",
                 model_version: str = "h9_dividend_runup_v1"):
        self._universe = universe
        self._timeframe = timeframe
        self._model_version = model_version

    def build_sleeve(self, as_of: str) -> dict:
        import pandas as pd  # lazy: heavy dep only on the live path
        sys.path.insert(0, str(REPO_ROOT / "ml"))
        from src.features.cross_sectional import load_panels  # type: ignore
        from src.service.dividend_sleeve import build_sleeve_signal, load_dividend_calendar  # type: ignore

        panel, _, _ = load_panels(self._universe, timeframe=self._timeframe)
        calendar = load_dividend_calendar()
        ts = pd.Timestamp(as_of[:19] if "T" in as_of else as_of, tz="Europe/Moscow") \
            if not str(as_of).endswith(("+03:00",)) else pd.Timestamp(as_of)
        sig = build_sleeve_signal(panel, calendar, ts, model_version=self._model_version)
        return contracts.validate(sig, "sleeve_signal")


class LiveCombiner:
    """The real risk_manager combiner with ML risk_analytics (H4 vol + H5 regime gate)."""

    def __init__(self, universe: list[str], timeframe: str = "1D", hedge_mode: str = "sector",
                 target_book_vol_annual: float = 0.12):
        self._universe = universe
        self._timeframe = timeframe
        self._hedge_mode = hedge_mode
        self._target_vol = target_book_vol_annual

    def combine(self, sleeve_signals: list[dict], as_of: str) -> dict:
        import pandas as pd  # lazy
        sys.path.insert(0, str(REPO_ROOT))
        sys.path.insert(0, str(REPO_ROOT / "ml"))
        from risk_manager.src import CombinerConfig, combine  # type: ignore
        from src.features.cross_sectional import load_panels  # type: ignore
        from src.service.risk_analytics import build_risk_analytics  # type: ignore

        panel, _, market = load_panels(self._universe, timeframe=self._timeframe)
        ts = pd.Timestamp(as_of) if str(as_of).endswith("+03:00") else \
            pd.Timestamp(str(as_of)[:19], tz="Europe/Moscow")
        risk_analytics = build_risk_analytics(panel, market, as_of=ts)
        cfg = CombinerConfig(hedge_mode=self._hedge_mode, timeframe=self._timeframe,
                             target_book_vol_annual=self._target_vol)
        book = combine(sleeve_signals, risk_analytics, cfg)
        return contracts.validate(book.to_dict(), "risk_book")


def _as_of_date(as_of: str):
    """Parse the YYYY-MM-DD prefix of an as_of timestamp to a date."""
    import datetime as _dt
    return _dt.date.fromisoformat(str(as_of)[:10])


class LiveBackend:
    """backend/data block — calls its public Python API in-process (read-only), preferred,
    or a configured CLI override.

    The backend block (backend.ingest / backend.integrity / backend.store) exposes plain
    functions and shares the project's MOEX trading calendar; for a single VDS this in-process
    handshake is lighter than HTTP. Set blocks.backend.{ingest_cmd,integrity_cmd,prices_cmd}
    to force a subprocess CLI instead (must emit the same JSON).
    """

    def __init__(self, cfg: dict):
        self._ingest = cfg.get("ingest_cmd")
        self._integrity = cfg.get("integrity_cmd")
        self._prices = cfg.get("prices_cmd")
        self._timeframe = cfg.get("prices_timeframe", "1D")

    def run_ingest(self, as_of: str) -> dict:
        if self._ingest:
            return _run_json_cmd([*self._ingest, "--date", str(as_of)[:10]])
        from backend.ingest import run_ingest as backend_ingest  # type: ignore
        return backend_ingest(today=_as_of_date(as_of))

    def integrity_gate(self, as_of: str) -> IntegrityStatus:
        if self._integrity:
            out = _run_json_cmd([*self._integrity, "--date", str(as_of)[:10]])
        else:
            from backend.integrity import run_checks  # type: ignore
            out = run_checks(ref=_as_of_date(as_of))
        return IntegrityStatus(status=out.get("status", "HALT"), as_of=as_of,
                               reasons=list(out.get("reasons", [])))

    def latest_prices(self, universe: list[str], as_of: str) -> dict[str, float]:
        if self._prices:
            return {k: float(v) for k, v in _run_json_cmd([*self._prices, "--date", str(as_of)[:10]]).items()}
        return _prices_from_store(universe, as_of, self._timeframe)


class LiveExecution:
    """execution block via its CLI (parallel chat). Configured command in blocks.execution.

    The command receives the reconciliation request (risk_book + current book + prices +
    capital + mode) as JSON on stdin and must return {orders, reports, positions, rejected}.
    Refuses to run if unconfigured — we never silently skip real execution.
    """

    def __init__(self, cfg: dict):
        self._command = cfg.get("command")

    def reconcile_and_execute(self, *, risk_book: dict, positions: list[dict],
                              prices: dict[str, float], capital: float, mode: str,
                              trade_date: str, phase: str) -> ExecutionResult:
        if not self._command:
            raise RuntimeError(
                "block_mode=live but blocks.execution.command is unset — the execution block "
                "CLI is not wired yet. Keep execution on the mock path until it lands.")
        req = {"risk_book": risk_book, "positions": positions, "prices": prices,
               "capital": capital, "mode": mode, "trade_date": trade_date, "phase": phase}
        out = _run_json_cmd([*self._command, "--mode", mode], payload=req)
        for rep in out.get("reports", []):
            contracts.validate(rep, "execution_report")
        for order in out.get("orders", []):
            contracts.validate(order, "order_request")
        return ExecutionResult(orders=out.get("orders", []), reports=out.get("reports", []),
                               positions=out.get("positions", []), rejected=out.get("rejected", []))


def _prices_from_store(universe: list[str], as_of: str, timeframe: str = "1D") -> dict[str, float]:
    """Last close at/<= as_of per instrument (names + sector/market hedge indices) via
    the backend parquet store (read-only)."""
    import pandas as pd  # lazy
    from backend import store  # type: ignore

    cutoff = pd.Timestamp(str(as_of)[:10])
    instruments = set(universe) | {"IMOEX", "MOEXFN", "MOEXOG", "MOEXMM", "MOEXCN", "MOEXTL"}
    out: dict[str, float] = {}
    for ticker in instruments:
        df = store.load_ticker(ticker, timeframe)
        if df is None or df.empty or "close" not in df.columns:
            continue
        begin = pd.to_datetime(df["begin"])
        if getattr(begin.dt, "tz", None) is not None:
            begin = begin.dt.tz_localize(None)
        sub = df[begin <= cutoff]
        if not sub.empty:
            out[ticker] = float(sub["close"].iloc[-1])
    return out
