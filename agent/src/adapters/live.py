"""Live block adapters — call the real blocks (read-only) and validate their JSON.

- LiveSleeve shells out to the ML serving CLI `ml/scripts/predict_dividend_sleeve.py --out -`
  (the process-boundary seam the ML block built for us) so NO pandas/ML internals enter the
  stdlib-only orchestrator core.
- LiveCombiner imports risk_manager's pure-Python combiner + ml risk_analytics (the same
  handshake risk_manager/scripts/demo_combine_h9.py uses) — read-only, lazy import.
- LiveBackend calls the backend block's functions in-process (ingest/integrity/store).
- LiveExecution drives the real execution `ExecutionEngine` in-process (discipline + dup-ledger
  + audit + paper broker), or a configured subprocess command if the block adds a JSON CLI.

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
    """The real ML H9 sleeve via its serving CLI seam — keeps pandas/ML out of the agent core.

    Shells out to `python ml/scripts/predict_dividend_sleeve.py --as-of <date> --out -`, which
    emits a validated `sleeve_signal` JSON on stdout. `command` defaults to running that script
    with the orchestrator's own interpreter (which carries the data deps on the VDS); override
    blocks.sleeve.command to pin a specific interpreter/path.
    """

    DEFAULT_SCRIPT = REPO_ROOT / "ml" / "scripts" / "predict_dividend_sleeve.py"

    def __init__(self, universe: list[str], timeframe: str = "1D",
                 model_version: str = "h9_dividend_runup_v1",
                 command: Optional[list[str]] = None):
        self._command = command or [sys.executable, str(self.DEFAULT_SCRIPT)]

    def build_sleeve(self, as_of: str) -> dict:
        sig = _run_json_cmd([*self._command, "--as-of", str(as_of)[:10], "--out", "-"])
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


def _resolve(path: str) -> Path:
    """Resolve a config path against the repo root unless it is already absolute."""
    p = Path(path)
    return p if p.is_absolute() else (REPO_ROOT / p)


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
    """The real execution block — drives `execution.src.engine.ExecutionEngine` in-process.

    The engine owns reconciliation, the H9 −12/−2 discipline check, lot rounding (MOEX lot
    sizes), the idempotent duplicate-order ledger, the paper broker (sim) and the audit log.
    The agent persists the resulting book + per-sleeve P&L. `mode` maps dry-run/paper/live;
    live is additionally gated inside execution (allow_live + EXECUTION_ALLOW_LIVE=1).

    If blocks.execution.command is set, a subprocess CLI is used instead (JSON request on
    stdin -> {orders, reports, positions, rejected}) for when execution adds such a CLI.
    """

    def __init__(self, cfg: dict):
        self._command = cfg.get("command")
        self._broker_backend = cfg.get("broker_backend", "sim")
        self._allow_live = bool(cfg.get("allow_live", False))
        self._state_dir = cfg.get("state_dir")
        self._audit_dir = cfg.get("audit_dir")

    def reconcile_and_execute(self, *, risk_book: dict, positions: list[dict],
                              prices: dict[str, float], capital: float, mode: str,
                              trade_date: str, phase: str) -> ExecutionResult:
        if self._command:
            return self._via_cli(risk_book, positions, prices, capital, mode, trade_date, phase)
        return self._via_engine(risk_book, positions, prices, capital, mode)

    def _via_engine(self, risk_book: dict, positions: list[dict], prices: dict[str, float],
                    capital: float, mode: str) -> ExecutionResult:
        from execution.src.config import ExecutionConfig, Mode  # type: ignore
        from execution.src.engine import ExecutionEngine  # type: ignore

        cfg = ExecutionConfig(mode=Mode(mode), broker_backend=self._broker_backend,
                              capital=capital, allow_live=self._allow_live)
        if self._state_dir:
            cfg.state_dir = _resolve(self._state_dir)
        if self._audit_dir:
            cfg.audit_dir = _resolve(self._audit_dir)
        engine = ExecutionEngine(cfg)

        snapshot = {"positions": [{"ticker": p["ticker"], "lots": int(p["lots"]), "avg_price": 0,
                                   "market_price": 0, "market_value": 0, "unrealized_pnl": 0}
                                  for p in positions]} if positions else None
        # on_critical='warn': the agent owns the kill-switch/halt; execution findings are surfaced,
        # not auto-killed (auto-kill would wedge future cycles via its KILL file).
        res = engine.run_cycle(risk_book, prices, current_positions=snapshot, on_critical="warn")
        for rep in res.reports:
            contracts.validate(rep, "execution_report")

        rejected = [{"ticker": s.get("instrument", "*"), "reason": s.get("reason", "")}
                    for s in res.skipped]
        rejected += [{"ticker": "*", "reason": f"discipline:{f.get('rule', f)}"} for f in res.findings]
        post_positions = ([] if mode == "dry-run"
                          else _enrich_book(engine.broker.positions(), risk_book, prices))
        return ExecutionResult(orders=res.submitted, reports=res.reports,
                               positions=post_positions, rejected=rejected)

    def _via_cli(self, risk_book, positions, prices, capital, mode, trade_date, phase) -> ExecutionResult:
        req = {"risk_book": risk_book, "positions": positions, "prices": prices,
               "capital": capital, "mode": mode, "trade_date": trade_date, "phase": phase}
        out = _run_json_cmd([*self._command, "--mode", mode], payload=req)
        for rep in out.get("reports", []):
            contracts.validate(rep, "execution_report")
        for order in out.get("orders", []):
            contracts.validate(order, "order_request")
        return ExecutionResult(orders=out.get("orders", []), reports=out.get("reports", []),
                               positions=out.get("positions", []), rejected=out.get("rejected", []))


def _enrich_book(broker_positions: dict[str, int], risk_book: dict,
                 prices: dict[str, float]) -> list[dict]:
    """Turn execution's signed lot map into the agent's position rows, re-attaching the sleeve
    attribution + hedge flag from the risk_book (execution does not track those)."""
    hedge = {leg["instrument"] for leg in risk_book.get("hedge", {}).get("legs", [])}
    sc = {p["ticker"]: p.get("sleeve_contributions", {}) for p in risk_book.get("net_positions", [])}
    out: list[dict] = []
    for ticker, lots in broker_positions.items():
        if not lots:
            continue
        price = prices.get(ticker)
        out.append({"ticker": ticker, "lots": int(lots),
                    "avg_price": round(price, 4) if price else 0.0,
                    "last_price": round(price, 4) if price else None,
                    "is_hedge": ticker in hedge, "sleeve_contributions": sc.get(ticker, {})})
    return out


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
