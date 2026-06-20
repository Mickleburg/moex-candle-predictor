"""ExecutionEngine — the EOD step-6 orchestrator.

Per cycle: refuse if halted or non-trading-day -> discipline-check the book -> reconcile to delta
LIMIT orders -> drop duplicates (idempotent client_order_id ledger) -> submit via the broker ->
audit every event. Plus a kill-switch (cancel everything + halt) and a season replay used to
reconcile a paper run against the sleeve backtest.

Safety posture: live is gated in the broker factory; a CRITICAL discipline finding (holding into the
ex-gap) halts the cycle by default; the duplicate ledger is persisted only outside dry-run.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .audit import AuditLog
from .brokers import BrokerAdapter, make_broker
from .config import ExecutionConfig, Mode
from .discipline import DisciplineChecker
from .reconcile import reconcile
from .trading_calendar import TradingCalendar, _as_date, default_trading_calendar


@dataclass
class CycleResult:
    as_of: str
    mode: str
    submitted: list[dict] = field(default_factory=list)    # order_request dicts actually sent
    reports: list[dict] = field(default_factory=list)      # execution_report dicts
    duplicates: list[str] = field(default_factory=list)    # client_order_ids skipped as duplicates
    noops: list[str] = field(default_factory=list)         # instruments already at target
    skipped: list[dict] = field(default_factory=list)      # e.g. missing price
    findings: list[dict] = field(default_factory=list)     # discipline findings
    halted: bool = False
    is_production: bool = False

    def summary_line(self) -> str:
        if self.halted:
            return f"[{self.as_of}] HALTED — no orders ({self.mode})"
        return (f"[{self.as_of}] {self.mode}: {len(self.submitted)} order(s), "
                f"{len(self.duplicates)} dup-skip, {len(self.noops)} no-op, "
                f"{len(self.skipped)} unpriced; is_production={self.is_production}")


def positions_from_snapshot(snapshot: dict | None) -> dict[str, int]:
    """Signed lot map from a `portfolio_snapshot` dict."""
    if not snapshot:
        return {}
    return {p["ticker"]: int(p["lots"]) for p in snapshot.get("positions", [])}


def _update_position(lots: int, avg: float, fill_signed: int, price: float) -> tuple[int, float]:
    """Apply a signed fill at ``price`` to a (lots, avg_price) position; return the new pair."""
    new_lots = lots + fill_signed
    if lots == 0 or (lots > 0) == (fill_signed > 0):          # opening or adding to the same side
        notional = abs(lots) * avg + abs(fill_signed) * price
        new_avg = notional / abs(new_lots) if new_lots != 0 else 0.0
    elif abs(fill_signed) <= abs(lots):                        # reducing the existing side
        new_avg = avg
    else:                                                      # crossing through zero
        new_avg = price
    return new_lots, (new_avg if new_lots != 0 else 0.0)


def _resulting_book(positions: list[dict], prices: dict[str, float], risk_book: dict,
                    result: "CycleResult") -> list[dict]:
    """Book AFTER fills: current positions + FILLED reports, enriched from the risk_book.

    Shape matches what the orchestrator's `_replace_book` expects
    ({ticker, lots, avg_price, last_price, is_hedge, sleeve_contributions}). PLACED (resting live
    limit) / DRY_RUN reports do not move the book — only FILLED (paper) does.
    """
    meta: dict[str, dict] = {}
    for p in risk_book.get("net_positions", []):
        meta[p["ticker"]] = {"is_hedge": False, "sector": p.get("sector"),
                             "sleeve_contributions": p.get("sleeve_contributions", {})}
    for leg in (risk_book.get("hedge") or {}).get("legs", []):
        meta[leg["instrument"]] = {"is_hedge": True, "sector": leg["instrument"],
                                   "sleeve_contributions": {}}

    book: dict[str, dict] = {}
    for p in positions:
        book[p["ticker"]] = {"lots": int(p.get("lots", 0)),
                             "avg_price": float(p.get("avg_price", 0.0) or 0.0)}

    for rep, order in zip(result.reports, result.submitted):
        if rep["status"] != "FILLED":
            continue
        t = rep["ticker"]
        filled = int(rep["filled_quantity_lots"])
        price = float(rep["avg_fill_price"] if rep["avg_fill_price"] is not None else order["limit_price"])
        signed = filled if order["side"] == "BUY" else -filled
        cur = book.get(t, {"lots": 0, "avg_price": 0.0})
        new_lots, new_avg = _update_position(cur["lots"], cur["avg_price"], signed, price)
        book[t] = {"lots": new_lots, "avg_price": new_avg}

    out: list[dict] = []
    for t, b in book.items():
        if b["lots"] == 0:
            continue
        m = meta.get(t, {})
        lp = prices.get(t)
        out.append({"ticker": t, "lots": int(b["lots"]), "avg_price": round(float(b["avg_price"]), 6),
                    "last_price": float(lp) if lp is not None else None,
                    "is_hedge": bool(m.get("is_hedge", False)),
                    "sleeve_contributions": m.get("sleeve_contributions", {})})
    return sorted(out, key=lambda x: x["ticker"])


def _ticker_from_coid(client_order_id: str) -> str:
    """Recover the ticker from a deterministic client_order_id (exec-DATE-TICKER-SIDE-QTY)."""
    parts = client_order_id.split("-")
    return parts[2] if len(parts) >= 5 else client_order_id


def _collect_rejected(result: "CycleResult") -> list[dict]:
    """[{ticker, reason}] for everything not turned into a live order this cycle."""
    rejected: list[dict] = []
    for sk in result.skipped:
        rejected.append({"ticker": sk.get("instrument"), "reason": sk.get("reason")})
    for coid in result.duplicates:
        rejected.append({"ticker": _ticker_from_coid(coid), "reason": "duplicate_intent"})
    for f in result.findings:
        if f.get("severity") == "critical":
            rejected.append({"ticker": f.get("instrument"), "reason": f.get("message")})
    return rejected


class ExecutionEngine:
    def __init__(
        self,
        config: ExecutionConfig | None = None,
        broker: BrokerAdapter | None = None,
        calendar: TradingCalendar | None = None,
        audit: AuditLog | None = None,
    ) -> None:
        self.config = config or ExecutionConfig()
        self.broker = broker or make_broker(self.config)
        self.calendar = calendar or default_trading_calendar()
        self.audit = audit or AuditLog(self.config.audit_dir, self.config.is_production)
        self.checker = DisciplineChecker(self.config, self.calendar)
        self.config.state_dir = Path(self.config.state_dir)
        self.config.state_dir.mkdir(parents=True, exist_ok=True)
        self._ledger_path = self.config.state_dir / "submitted.txt"
        self._kill_path = self.config.state_dir / "KILL"
        self._submitted: set[str] = self._load_ledger()

    # --- persistence ----------------------------------------------------------------
    def _load_ledger(self) -> set[str]:
        if self.config.mode is Mode.DRY_RUN or not self._ledger_path.exists():
            return set()
        return {ln.strip() for ln in self._ledger_path.read_text(encoding="utf-8").splitlines() if ln.strip()}

    def _remember(self, client_order_id: str) -> None:
        self._submitted.add(client_order_id)
        if self.config.mode is not Mode.DRY_RUN:
            with self._ledger_path.open("a", encoding="utf-8") as handle:
                handle.write(client_order_id + "\n")

    @property
    def halted(self) -> bool:
        return self._kill_path.exists()

    # --- kill switch ----------------------------------------------------------------
    def kill(self, reason: str = "manual") -> list[dict]:
        """Cancel every open order and halt. Idempotent."""
        reports = self.broker.cancel_all()
        self._kill_path.write_text(f"{datetime.now(timezone.utc).isoformat()} {reason}\n", encoding="utf-8")
        self.audit.record("kill_switch", {"reason": reason, "canceled": reports})
        return reports

    def reset_kill(self) -> None:
        if self._kill_path.exists():
            self._kill_path.unlink()
        self.audit.record("kill_reset", {})

    # --- main cycle -----------------------------------------------------------------
    def run_cycle(
        self,
        risk_book: dict,
        prices: dict[str, float],
        current_positions: dict | None = None,
        anchors: dict[str, object] | None = None,
        on_critical: str = "halt",
    ) -> CycleResult:
        as_of = str(risk_book.get("as_of", ""))
        result = CycleResult(as_of=as_of, mode=self.config.mode.value,
                             is_production=self.config.is_production)

        if self.halted:
            result.halted = True
            self.audit.record("cycle_skipped", {"as_of": as_of, "reason": "halted"})
            return result

        if not self.calendar.is_trading_day(_as_date(as_of) if as_of else datetime.now(timezone.utc).date()):
            result.skipped.append({"instrument": "*", "reason": "non_trading_day"})
            self.audit.record("cycle_skipped", {"as_of": as_of, "reason": "non_trading_day"})
            return result

        findings = self.checker.check_book(risk_book, anchors)
        result.findings = [asdict(f) for f in findings]
        if findings:
            self.audit.record("discipline", {"as_of": as_of, "findings": result.findings})
        if DisciplineChecker.has_critical(findings) and on_critical == "halt":
            self.audit.record("discipline_halt", {"as_of": as_of, "findings": result.findings})
            self.kill(reason="discipline_critical")
            result.halted = True
            return result

        current_lots = (positions_from_snapshot(current_positions)
                        if current_positions is not None else self.broker.positions())
        recon = reconcile(risk_book, prices, current_lots, self.config)
        result.noops = recon.noops
        result.skipped.extend(recon.skipped)

        for order in recon.orders:
            req = order.to_order_request()
            coid = req["client_order_id"]
            if coid in self._submitted:
                result.duplicates.append(coid)
                self.audit.record("duplicate_skipped", {"order": req})
                continue
            self.audit.record("order_submitted", {"order": req, "binding": order.binding})
            report = self.broker.place_order(req)
            result.submitted.append(req)
            result.reports.append(report)
            self.audit.record("execution_report", {"report": report})
            if report["status"] != "REJECTED":
                self._remember(coid)

        self.audit.record("cycle_done", {"as_of": as_of, "summary": result.summary_line()})
        return result

    # --- orchestrator seam ----------------------------------------------------------
    def reconcile_and_execute(
        self,
        *,
        risk_book: dict,
        positions: list[dict],
        prices: dict[str, float],
        anchors: dict[str, object] | None = None,
        on_critical: str = "halt",
    ) -> dict:
        """One EOD reconciliation+execution, in the envelope the agent orchestrator consumes.

        Mirrors `agent/src/adapters/live.py::LiveExecution.reconcile_and_execute`: takes the target
        `risk_book`, the current book (`positions`), latest `prices` and the already-configured
        capital/mode, and returns ``{orders, reports, positions, rejected}`` — orders are
        `order_request`s, reports are `execution_report`s, positions is the book AFTER fills, rejected
        is ``[{ticker, reason}]``. This is what the `serve` CLI wraps with stdin/stdout JSON.
        """
        current = {"positions": [{"ticker": p["ticker"], "lots": int(p.get("lots", 0))}
                                 for p in positions]}
        res = self.run_cycle(risk_book, prices, current_positions=current,
                             anchors=anchors, on_critical=on_critical)
        return {
            "orders": res.submitted,
            "reports": res.reports,
            "positions": _resulting_book(positions, prices, risk_book, res),
            "rejected": _collect_rejected(res),
            "halted": res.halted,
            "is_production": res.is_production,
        }

    # --- season replay --------------------------------------------------------------
    def run_season(
        self,
        days: list[dict],
        anchors: dict[str, object] | None = None,
        on_critical: str = "warn",
    ) -> dict:
        """Replay a sequence of EOD cycles, threading positions through the broker.

        Each ``days`` item is ``{"risk_book": {...}, "prices": {...}}``. Positions come from the
        broker after each fill (PaperBroker), so this traces holdings exactly as the daily
        reconciliation would build them — the artifact compared against the sleeve backtest.
        Returns {cycles, held_by_day, final_positions}.
        """
        cycles: list[CycleResult] = []
        held_by_day: dict[str, dict[str, int]] = {}
        for item in days:
            book = item["risk_book"]
            res = self.run_cycle(book, item["prices"], anchors=anchors, on_critical=on_critical)
            cycles.append(res)
            held_by_day[res.as_of] = self.broker.positions()
        return {"cycles": cycles, "held_by_day": held_by_day,
                "final_positions": self.broker.positions()}
