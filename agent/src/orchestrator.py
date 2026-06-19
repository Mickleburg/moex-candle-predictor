"""The V3 daily operational cycle — the state machine that glues the blocks together.

EOD (after clearing, ~19:05 MSK):
    1 ingest  ->  3 integrity gate (HALT => no trading)  ->  4 ML sleeve(as_of)
    ->  5 risk_manager combiner (vol-target x H5 regime gate x limits x hedge)
    ->  6 execution reconcile (paper)  ->  7 persist (book + per-sleeve P&L)  ->  8 alert digest.
Pre-open (~09:30 MSK): kill-switch, overnight gap/HALT check, confirm/cancel limit orders.

Reactions wired in: data HALT -> do not trade; H5 regime gate -> the book's gross is already
cut by the combiner and the agent executes that smaller book; a new ex-date enters via the
sleeve at EOD; kill-switch -> trading stops, monitoring continues. Idempotent per
(trade_date, phase) and recoverable after a restart (all state is in the SQLite store).
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

from . import contracts, pnl
from . import trading_calendar as tcal
from .adapters import build_adapters
from .adapters.registry import Adapters
from .config import AgentConfig, load_config
from .notifier import Notifier, build_notifier
from .state_store import StateStore

MSK_OFFSET = "+03:00"


class Orchestrator:
    def __init__(self, config: Optional[AgentConfig] = None, *, store: Optional[StateStore] = None,
                 adapters: Optional[Adapters] = None, notifier: Optional[Notifier] = None):
        self.config = config or load_config()
        self.config.ensure_dirs()
        self.store = store or StateStore(self.config.state_db)
        self.adapters = adapters or build_adapters(self.config)
        self.notifier = notifier or build_notifier(self.config)

    # ----------------------------------------------------------------- EOD cycle
    def run_eod_cycle(self, trade_date: Optional[str] = None, as_of: Optional[str] = None,
                      *, force: bool = False) -> dict[str, Any]:
        td = trade_date or date.today().isoformat()
        as_of = as_of or f"{td}T19:05:00{MSK_OFFSET}"
        phase = "eod"

        if not tcal.is_trading_day(td) and not force:
            return {"status": "skipped", "reason": "not a trading day", "trade_date": td}

        claim = self.store.begin_cycle(td, phase, mode=self.config.mode,
                                       block_mode=self.config.block_mode, as_of=as_of, force=force)
        if not claim["started"]:
            return {"status": "skipped_idempotent", "trade_date": td,
                    "prior_status": claim["prior"]["status"], "result": self._prior_result(claim)}

        try:
            return self._eod_body(td, as_of, phase)
        except Exception as exc:  # noqa: BLE001 - never leave a cycle 'running'
            self.store.finish_cycle(td, phase, "failed", halt_reason=f"{type(exc).__name__}: {exc}")
            self._alert("EOD cycle FAILED", f"trade_date={td}\nerror: {type(exc).__name__}: {exc}")
            raise

    def _eod_body(self, td: str, as_of: str, phase: str) -> dict[str, Any]:
        kill = self.store.kill_switch_engaged()

        # step 1 — ingest
        ingest = self.adapters.backend.run_ingest(as_of)

        # step 3 — integrity gate (HALT => no trading)
        integrity = self.adapters.backend.integrity_gate(as_of)
        if not integrity.ok:
            result = self._cycle_result(td, as_of, mode="paper", orders=[], rejected=[],
                                        risk_summary={"halt": True, "integrity_reasons": integrity.reasons,
                                                      "kill_switch": kill})
            self._persist_result(td, phase, "halted", result)
            self._alert("DATA HALT — not trading",
                        f"trade_date={td}\nreasons: {', '.join(integrity.reasons)}")
            return {"status": "halted", "trade_date": td, "result": result}

        # step 4 — ML sleeve
        sleeve_signal = contracts.validate(self.adapters.sleeve.build_sleeve(as_of), "sleeve_signal")

        # step 5 — risk_manager combiner
        risk_book = contracts.validate(self.adapters.combiner.combine([sleeve_signal], as_of), "risk_book")
        rs = risk_book["risk_scalars"]

        # prices for sizing + marks
        prices = self.adapters.backend.latest_prices(self.config.universe, as_of)

        # step 6 — execution (paper-first: live only with the hard gate)
        eff_mode, mode_note = self._effective_exec_mode()
        if kill:
            exec_orders, exec_reports, exec_positions, rejected = [], [], None, []
            exec_note = "kill-switch engaged — execution skipped, monitoring only"
        else:
            er = self.adapters.execution.reconcile_and_execute(
                risk_book=risk_book, positions=self.store.get_positions(), prices=prices,
                capital=self.config.capital_rub, mode=eff_mode, trade_date=td, phase=phase)
            exec_orders, exec_reports, exec_positions, rejected = er.orders, er.reports, er.positions, er.rejected
            exec_note = f"executed {len(exec_orders)} delta-order(s) in {eff_mode} mode. {mode_note}".strip()

        # step 7 — persist book + per-sleeve P&L attribution + shadow log
        self._record_orders_and_fills(td, phase, exec_orders, exec_reports)
        if not kill and eff_mode in ("paper", "live") and exec_positions is not None:
            self._replace_book(exec_positions)
        book = self.store.get_positions()
        sleeve_pnl = pnl.attribute_book_pnl(book)
        for sleeve, vals in sleeve_pnl.items():
            self.store.record_pnl_attribution(td, sleeve, realized=0.0,
                                               unrealized=vals["unrealized"], gross=vals["gross"])
        pnl.append_shadow_log(self.config.shadow_log, trade_date=td, as_of=as_of,
                              risk_book=risk_book, positions=book, sleeve_pnl=sleeve_pnl)

        # step 8 — result + digest
        risk_summary = {
            "halt": False, "kill_switch": kill,
            "exposure_scalar": rs.get("exposure_scalar"), "regime_novel": rs.get("regime_novel"),
            "vol_scalar": rs.get("vol_scalar"),
            "directional_gross": rs.get("directional_gross"), "total_gross": rs.get("total_gross"),
            "binding_limits": risk_book["limits"].get("binding", []),
            "hedge_mode": risk_book["hedge"]["mode"],
            "capital_rub": self.config.capital_rub,
            "block_modes": self.adapters.modes, "calendar": tcal.calendar_source(),
            "exec_note": exec_note, "sleeve_pnl": sleeve_pnl,
        }
        result = self._cycle_result(td, as_of, mode=eff_mode, orders=exec_orders,
                                    rejected=rejected, risk_summary=risk_summary)
        status = "killed" if kill else "completed"
        self._persist_result(td, phase, status, result)
        self._alert(f"EOD digest — {td} ({status})", self._digest_text(td, result, risk_book, sleeve_pnl))
        return {"status": status, "trade_date": td, "result": result}

    # ----------------------------------------------------------------- pre-open
    def run_preopen(self, trade_date: Optional[str] = None, *, force: bool = False) -> dict[str, Any]:
        td = trade_date or date.today().isoformat()
        as_of = f"{td}T09:30:00{MSK_OFFSET}"
        phase = "preopen"
        if not tcal.is_trading_day(td) and not force:
            return {"status": "skipped", "reason": "not a trading day", "trade_date": td}

        claim = self.store.begin_cycle(td, phase, mode=self.config.mode,
                                       block_mode=self.config.block_mode, as_of=as_of, force=force)
        if not claim["started"]:
            return {"status": "skipped_idempotent", "trade_date": td}

        try:
            open_orders = self.store.open_orders()
            if self.store.kill_switch_engaged():
                self._cancel_all(open_orders)
                self._persist_result(td, phase, "killed",
                                     {"as_of": as_of, "mode": "paper", "evaluated_tickers": self.config.universe,
                                      "selected_orders": [], "rejected_candidates": [],
                                      "risk_summary": {"kill_switch": True, "canceled": len(open_orders)}})
                self._alert("PRE-OPEN — kill-switch", f"trade_date={td}\ncanceled {len(open_orders)} open order(s)")
                return {"status": "killed", "trade_date": td}

            integrity = self.adapters.backend.integrity_gate(as_of)
            if not integrity.ok:
                self._cancel_all(open_orders)
                self._persist_result(td, phase, "halted",
                                     {"as_of": as_of, "mode": "paper", "evaluated_tickers": self.config.universe,
                                      "selected_orders": [], "rejected_candidates": [],
                                      "risk_summary": {"halt": True, "integrity_reasons": integrity.reasons,
                                                       "canceled": len(open_orders)}})
                self._alert("PRE-OPEN — overnight HALT",
                            f"trade_date={td}\nreasons: {', '.join(integrity.reasons)}\ncanceled {len(open_orders)} order(s)")
                return {"status": "halted", "trade_date": td}

            # healthy: confirm open limit orders stand
            result = {"as_of": as_of, "mode": "paper", "evaluated_tickers": self.config.universe,
                      "selected_orders": [], "rejected_candidates": [],
                      "risk_summary": {"confirmed_open_orders": len(open_orders),
                                       "kill_switch": False, "calendar": tcal.calendar_source()}}
            self._persist_result(td, phase, "completed", result)
            self._alert(f"PRE-OPEN OK — {td}", f"confirmed {len(open_orders)} open order(s); no overnight HALT")
            return {"status": "completed", "trade_date": td, "result": result}
        except Exception as exc:  # noqa: BLE001
            self.store.finish_cycle(td, phase, "failed", halt_reason=f"{type(exc).__name__}: {exc}")
            raise

    # ----------------------------------------------------------------- helpers
    def _effective_exec_mode(self) -> tuple[str, str]:
        if self.config.mode == "live" and not self.config.live_enabled():
            return "paper", "live requested but enable_live gate is off -> forced paper (paper-first)."
        return self.config.mode, ""

    def _record_orders_and_fills(self, td: str, phase: str, orders: list[dict], reports: list[dict]) -> None:
        report_status = {r["client_order_id"]: r["status"] for r in reports}
        for order in orders:
            coid = order["client_order_id"]
            if self.store.order_exists(coid):     # dedup guard (idempotency 2nd line of defence)
                continue
            self.store.record_order(order, trade_date=td, phase=phase,
                                    status=report_status.get(coid, "PLACED"))
        for rep in reports:
            self.store.record_execution(rep)

    def _replace_book(self, new_positions: list[dict]) -> None:
        new_by_ticker = {p["ticker"]: p for p in new_positions}
        for p in self.store.get_positions():
            if p["ticker"] not in new_by_ticker:
                self.store.upsert_position(p["ticker"], 0, 0.0, None)   # closed
        for p in new_positions:
            self.store.upsert_position(p["ticker"], int(p["lots"]), float(p.get("avg_price", 0.0)),
                                       p.get("last_price"), is_hedge=bool(p.get("is_hedge")),
                                       sleeve_contributions=p.get("sleeve_contributions") or {})

    def _cancel_all(self, open_orders: list[dict]) -> None:
        for o in open_orders:
            self.store.set_order_status(o["client_order_id"], "CANCELED")

    def _cycle_result(self, td: str, as_of: str, *, mode: str, orders: list[dict],
                      rejected: list[dict], risk_summary: dict) -> dict:
        result = {
            "as_of": as_of, "mode": mode, "evaluated_tickers": list(self.config.universe),
            "selected_orders": orders, "rejected_candidates": rejected, "risk_summary": risk_summary,
        }
        return contracts.validate(result, "agent_cycle_result")

    def _persist_result(self, td: str, phase: str, status: str, result: dict) -> None:
        out = Path(self.config.cycle_results_dir) / f"{td}_{phase}.json"
        import json
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        halt_reason = None
        if status in ("halted", "killed", "failed"):
            reasons = result.get("risk_summary", {}).get("integrity_reasons")
            halt_reason = ", ".join(reasons) if isinstance(reasons, list) else (reasons or status)
        self.store.finish_cycle(td, phase, status, result=result, halt_reason=halt_reason)
        self.store.set_flag("last_cycle", {"trade_date": td, "phase": phase, "status": status,
                                           "at": datetime.now().astimezone().isoformat()})

    def _prior_result(self, claim: dict) -> Optional[dict]:
        prior = claim.get("prior") or {}
        rj = prior.get("result_json")
        if rj:
            import json
            return json.loads(rj)
        return None

    def _alert(self, subject: str, body: str) -> None:
        self.notifier.send(subject, body)

    @staticmethod
    def _digest_text(td: str, result: dict, risk_book: dict, sleeve_pnl: dict) -> str:
        rs = result["risk_summary"]
        lines = [f"trade_date={td}  mode={result['mode']}",
                 f"orders: {len(result['selected_orders'])}  "
                 f"gross(dir/total)={rs.get('directional_gross')}/{rs.get('total_gross')}",
                 f"regime: exposure_scalar={rs.get('exposure_scalar')} novel={rs.get('regime_novel')}",
                 f"hedge={rs.get('hedge_mode')}  binding={rs.get('binding_limits')}",
                 f"kill_switch={rs.get('kill_switch')}  calendar={rs.get('calendar')}"]
        for sleeve, vals in sleeve_pnl.items():
            lines.append(f"  P&L[{sleeve}]: unrealized={vals['unrealized']:.2f} gross={vals['gross']:.2f}")
        for o in result["selected_orders"][:12]:
            lines.append(f"  {o['side']} {o['quantity_lots']} {o['ticker']} @ {o.get('limit_price')}")
        return "\n".join(lines)
