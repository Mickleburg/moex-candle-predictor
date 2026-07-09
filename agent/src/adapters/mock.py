"""Mock block adapters — let the full daily cycle run end-to-end with stdlib alone.

These stand in for the parallel-chat blocks (backend ingest/integrity, execution paper
broker) and for the ML/risk_manager blocks when their data/deps are absent. They emit the
SAME JSON contracts the live blocks do, so the orchestrator code path is identical; only the
wiring (build_adapters) differs. Fault injection (HALT, regime gate, capacity) is exposed so
the orchestrator's reactions can be tested deterministically.
"""

from __future__ import annotations

from typing import Optional

from .base import ExecutionResult, IntegrityStatus

# Minimal static sector map (reference data) so the mock risk_book carries sectors like the
# live combiner. Duplicated on purpose: the mock must not import risk_manager.
_SECTOR = {
    "SBER": "MOEXFN", "VTBR": "MOEXFN",
    "GAZP": "MOEXOG", "LKOH": "MOEXOG", "ROSN": "MOEXOG", "NVTK": "MOEXOG",
    "TATN": "MOEXOG", "SNGS": "MOEXOG",
    "GMKN": "MOEXMM", "CHMF": "MOEXMM", "ALRS": "MOEXMM", "MAGN": "MOEXMM",
    "NLMK": "MOEXMM", "PLZL": "MOEXMM",
    "MGNT": "MOEXCN", "MTSS": "MOEXTL",
}


def _synthetic_price(ticker: str) -> float:
    """Deterministic, stable per-ticker price so marks/P&L are reproducible across runs."""
    return 50.0 + (abs(hash(ticker)) % 4000) / 10.0


class MockBackend:
    """Stands in for backend/data. Healthy by default; can inject a HALT for the gate test."""

    def __init__(self, *, halt: bool = False, halt_reasons: Optional[list[str]] = None,
                 universe: Optional[list[str]] = None):
        self._halt = halt
        self._halt_reasons = halt_reasons or ["injected integrity fault"]
        self._universe = universe or []

    def run_ingest(self, as_of: str) -> dict:
        return {"status": "ok", "as_of": as_of, "fetched": len(self._universe), "source": "mock"}

    def integrity_gate(self, as_of: str) -> IntegrityStatus:
        if self._halt:
            return IntegrityStatus(status="HALT", as_of=as_of, reasons=list(self._halt_reasons))
        return IntegrityStatus(status="OK", as_of=as_of, reasons=[])

    def latest_prices(self, universe: list[str], as_of: str) -> dict[str, float]:
        names = set(universe) | set(_SECTOR.values()) | {"IMOEX"}
        return {t: _synthetic_price(t) for t in names}


class MockSleeve:
    """Stands in for the ML H9 sleeve. Emits a valid sleeve_signal (long names + hedge rec)."""

    def __init__(self, *, longs: Optional[dict[str, float]] = None, abstain: bool = False,
                 model_version: str = "h9_dividend_runup_v1_mock", is_production: bool = False):
        # default: 3 names in their pre-ex window, capped inverse-vol-ish weights summing to ~1.
        self._longs = longs if longs is not None else {"SBER": 0.34, "LKOH": 0.34, "TATN": 0.32}
        self._abstain = abstain
        self._model_version = model_version
        # is_production=false by default (H9 is shadow until forward-gate + sign-off). Tests set
        # True to exercise the LIVE-capital path (a hypothetical signed-off sleeve).
        self._is_production = is_production

    def build_sleeve(self, as_of: str) -> dict:
        positions = [] if self._abstain else [
            {"ticker": t, "weight": round(w, 4), "leg": "long"}
            for t, w in sorted(self._longs.items(), key=lambda kv: -kv[1])
        ]
        gross_long = round(sum(self._longs.values()), 4) if not self._abstain else 0.0
        return {
            "sleeve": "s3_event",
            "strategy": "dividend_runup",
            "as_of": as_of,
            "market_neutral": True,
            "positions": positions,
            "hedge_recommendation": {"method": "sector_index", "fallback": "imoex_beta_adjusted",
                                     "notional": gross_long},
            "gross_long": gross_long,
            "model_version": self._model_version,
            "is_production": self._is_production,
        }


class MockCombiner:
    """Stands in for the risk_manager combiner: the risk-layer knobs the orchestrator reacts to
    (regime gate, vol scalar, caps) PLUS the shadow gate (invariants #9/#4). A sleeve gets LIVE
    capital only if it passed its gate (is_production=true AND a non-negative forward gate via
    `sleeve_status`); otherwise it goes to `shadow_positions` (0 live capital). Mirrors the real
    combiner's net_positions vs shadow_positions split so the agent's live/shadow attribution and
    forward-P&L demotion can be tested offline."""

    def __init__(self, *, hedge_mode: str = "market", exposure_scalar: float = 1.0,
                 vol_scalar: float = 1.0, regime_novel: bool = False,
                 max_name_weight: float = 0.34, max_gross: float = 1.0,
                 timeframe: str = "1D"):
        self._hedge_mode = hedge_mode
        self._exposure_scalar = exposure_scalar
        self._vol_scalar = vol_scalar
        self._regime_novel = regime_novel
        self._max_name = max_name_weight
        self._max_gross = max_gross
        self._timeframe = timeframe

    @staticmethod
    def _gate(sig: dict, status: dict | None) -> tuple[str, str, str]:
        """Classify a sleeve live/shadow (mirror of risk_manager._shadow_gate)."""
        if not sig.get("is_production", False):
            return ("shadow", "NOT_MET", "is_production=false")
        if status:
            if str(status.get("gate", "")).upper() == "NOT_MET":
                return ("shadow", "NOT_MET", str(status.get("reason", "forward gate not met")))
            fpnl = status.get("forward_pnl")
            if fpnl is not None and float(fpnl) < 0:
                return ("shadow", "NOT_MET", f"forward_pnl={fpnl}<0")
        return ("live", "MET", "production + forward gate met")

    def combine(self, sleeve_signals: list[dict], as_of: str,
                *, sleeve_status: dict[str, dict] | None = None) -> dict:
        gating, live_sigs, shadow_sigs = [], [], []
        for sig in sleeve_signals:
            state, gate, reason = self._gate(sig, (sleeve_status or {}).get(sig.get("sleeve")))
            gating.append({"sleeve": sig.get("sleeve"), "strategy": sig.get("strategy", ""),
                           "capital_state": state, "is_production": sig.get("is_production", False),
                           "gate": gate, "reason": reason})
            (live_sigs if state == "live" else shadow_sigs).append(sig)

        net_positions, hedge, live_gross, hedge_gross, binding = self._book(live_sigs)
        shadow_positions, shadow_hedge, shadow_gross, shadow_hedge_gross, _ = self._book(shadow_sigs)

        return {
            "as_of": as_of,
            "timeframe": self._timeframe,
            "sleeves": [{"sleeve": s["sleeve"], "strategy": s.get("strategy", ""),
                         "gross": round(sum(abs(float(p["weight"])) for p in s.get("positions", [])), 6)}
                        for s in sleeve_signals],
            "gating": gating,
            "net_positions": net_positions,
            "hedge": hedge,
            "shadow_positions": shadow_positions,
            "shadow_hedge": shadow_hedge,
            "risk_scalars": {
                "target_book_vol_annual": 0.12,
                "book_vol_estimate_annual": 0.0,
                "vol_scalar": round(self._vol_scalar, 6),
                "exposure_scalar": round(self._exposure_scalar, 6),
                "regime_novel": bool(self._regime_novel),
                "directional_gross": round(live_gross, 6),
                "total_gross": round(live_gross + hedge_gross, 6),
                "shadow_gross": round(shadow_gross, 6),
                "shadow_total_gross": round(shadow_gross + shadow_hedge_gross, 6),
            },
            "limits": {
                "max_name_weight": self._max_name,
                "max_sector_gross": 0.6,
                "max_gross": self._max_gross,
                "name_caps_ok": True,
                "sector_caps_ok": True,
                "gross_cap_ok": live_gross <= self._max_gross + 1e-6,
                "binding": sorted(set(binding)),
            },
            "model_version": "risk_combiner_v0_mock",
            # LIVE book is production only if there IS a live sleeve and all live sleeves are production.
            "is_production": bool(live_sigs) and all(s.get("is_production", False) for s in live_sigs),
        }

    def _book(self, sigs: list[dict]) -> tuple[list[dict], dict, float, float, list[str]]:
        """Net + cap + scale + hedge a set of sleeves into a sized book (live or shadow)."""
        net: dict[str, float] = {}
        contrib: dict[str, dict[str, float]] = {}
        for sig in sigs:
            for p in sig.get("positions", []):
                if p.get("leg") not in ("long", "short"):
                    continue
                w = float(p["weight"])
                net[p["ticker"]] = net.get(p["ticker"], 0.0) + w
                contrib.setdefault(p["ticker"], {})[sig["sleeve"]] = \
                    contrib.get(p["ticker"], {}).get(sig["sleeve"], 0.0) + w

        binding: list[str] = []
        capped = {}
        for t, w in net.items():
            if abs(w) > self._max_name + 1e-9:
                binding.append(f"name_cap:{t}")
                capped[t] = self._max_name if w > 0 else -self._max_name
            else:
                capped[t] = w
        book_scalar = self._vol_scalar * self._exposure_scalar
        if abs(self._vol_scalar - 1.0) > 1e-9 and capped:
            binding.append("vol_target")
        if self._exposure_scalar < 1.0 - 1e-9 and capped:
            binding.append("regime_gate")
        scaled = {t: w * book_scalar for t, w in capped.items()}
        gross = sum(abs(w) for w in scaled.values())
        if gross > self._max_gross + 1e-9:
            binding.append("gross_cap")
            s = self._max_gross / gross
            scaled = {t: w * s for t, w in scaled.items()}
        scaled = {t: w for t, w in scaled.items() if abs(w) > 1e-9}
        directional_gross = sum(abs(w) for w in scaled.values())

        positions = [
            {"ticker": t, "weight": round(w, 6), "side": "LONG" if w > 0 else "SHORT",
             "sector": _SECTOR.get(t, "IMOEX"),
             "sleeve_contributions": {k: round(v, 6) for k, v in contrib.get(t, {}).items()}}
            for t, w in sorted(scaled.items(), key=lambda kv: -abs(kv[1]))
        ]
        hedge = self._build_hedge(scaled)
        hedge_gross = sum(abs(leg["weight"]) for leg in hedge["legs"])
        return positions, hedge, directional_gross, hedge_gross, binding

    def _build_hedge(self, weights: dict[str, float]) -> dict:
        if not weights or self._hedge_mode == "none":
            return {"mode": "none", "legs": []}
        if self._hedge_mode == "sector":
            by_sec: dict[str, float] = {}
            for t, w in weights.items():
                by_sec[_SECTOR.get(t, "IMOEX")] = by_sec.get(_SECTOR.get(t, "IMOEX"), 0.0) + w
            legs = [{"instrument": s, "weight": round(-n, 6)} for s, n in sorted(by_sec.items())
                    if abs(n) > 1e-9]
            return {"mode": "sector", "legs": legs}
        net = sum(weights.values())
        legs = [{"instrument": "IMOEX", "weight": round(-net, 6)}] if abs(net) > 1e-9 else []
        return {"mode": "market", "legs": legs}


class PaperBrokerExecution:
    """Stands in for the execution block: a deterministic, TRACK-AWARE paper broker.

    Mirrors execution.src.reconcile's contract: the live (net_positions + hedge) and shadow
    (shadow_positions + shadow_hedge) books are reconciled SEPARATELY against their own current
    holdings — never netted, so a shadow short can't collapse a live long on the same ticker (2b/3e).
    Orders carry the track in the client_order_id (exec-DATE-TRACK-TICKER-SIDE-QTY) and returned
    positions carry a `track` tag, the same contract the real engine emits, so the orchestrator
    routes results by track. live trades the live track ONLY; dry-run/paper also paper-trade shadow.
    Lot size 1; dry-run computes orders without moving the book. Current holdings are read per track
    from each position's `track` tag (default live).
    """

    TRACKS = ("live", "shadow")

    def reconcile_and_execute(self, *, risk_book: dict, positions: list[dict],
                              prices: dict[str, float], capital: float, mode: str,
                              trade_date: str, phase: str) -> ExecutionResult:
        from execution.src.engine import _update_position  # type: ignore

        include_shadow = mode != "live"            # live trades net only; paper/dry-run fold shadow
        tracks = self.TRACKS if include_shadow else ("live",)
        tag = str(trade_date).replace("-", "")

        current: dict[str, dict[str, int]] = {"live": {}, "shadow": {}}
        current_avg: dict[tuple[str, str], float] = {}
        for p in positions:
            track = p.get("track", "live")
            current.setdefault(track, {})[p["ticker"]] = int(p["lots"])
            current_avg[(track, p["ticker"])] = float(p.get("avg_price", 0.0) or 0.0)

        orders, reports, new_positions, rejected = [], [], [], []
        for track in tracks:
            targets: dict[str, dict] = {}
            for inst, weight, is_hedge, sc in self._track_rows(risk_book, track):
                self._add_target(targets, rejected, inst, weight, prices, capital,
                                 is_hedge=is_hedge, sleeve_contributions=sc)
            cur = current.get(track, {})
            for ticker in sorted(set(targets) | set(cur)):
                target_lots = targets.get(ticker, {}).get("lots", 0)
                cur_lots = cur.get(ticker, 0)
                cur_avg = current_avg.get((track, ticker), 0.0)
                delta = target_lots - cur_lots
                price = prices.get(ticker)
                meta = targets.get(ticker, {})
                if delta != 0 and price is not None:
                    side = "BUY" if delta > 0 else "SELL"
                    coid = f"exec-{tag}-{track}-{ticker}-{side}-{abs(delta)}"
                    orders.append({"ticker": ticker, "side": side, "quantity_lots": abs(delta),
                                   "order_type": "LIMIT", "limit_price": round(price, 4),
                                   "client_order_id": coid})
                    status = "DRY_RUN" if mode == "dry-run" else "FILLED"
                    reports.append({"client_order_id": coid, "ticker": ticker, "status": status,
                                    "exchange_order_id": None if mode == "dry-run" else f"mock-{coid}",
                                    "filled_quantity_lots": 0 if mode == "dry-run" else abs(delta),
                                    "avg_fill_price": None if mode == "dry-run" else round(price, 4),
                                    "message": f"paper {status.lower()} {side} {abs(delta)} {ticker} [{track}]"})

                # resulting book: dry-run keeps current; paper moves to target. avg_price is the
                # carried entry cost basis (weighted on fills), NOT the current mark — otherwise
                # unrealized P&L reads 0 every cycle and the invariant #9 forward-P&L gate is dead.
                final_lots = cur_lots if mode == "dry-run" else target_lots
                if mode != "dry-run" and delta != 0 and price is not None:
                    _, final_avg = _update_position(cur_lots, cur_avg, delta, price)
                else:
                    final_avg = cur_avg
                if final_lots != 0:
                    new_positions.append({
                        "ticker": ticker, "lots": final_lots,
                        "avg_price": round(final_avg, 6),
                        "last_price": round(price, 4) if price is not None else None,
                        "is_hedge": meta.get("is_hedge", False),
                        "sleeve_contributions": meta.get("sleeve_contributions", {}),
                        "track": track,
                    })

        return ExecutionResult(orders=orders, reports=reports,
                               positions=new_positions, rejected=rejected)

    @staticmethod
    def _track_rows(risk_book: dict, track: str) -> list[tuple[str, float, bool, dict]]:
        """(instrument, weight, is_hedge, sleeve_contributions) rows for ONE track's book."""
        if track == "live":
            names, hedge = risk_book.get("net_positions", []), risk_book.get("hedge") or {}
        else:
            names, hedge = risk_book.get("shadow_positions", []), risk_book.get("shadow_hedge") or {}
        rows = [(p["ticker"], float(p["weight"]), False, p.get("sleeve_contributions", {}))
                for p in names]
        if hedge.get("mode") not in (None, "none"):
            rows += [(leg["instrument"], float(leg["weight"]), True, {}) for leg in hedge.get("legs", [])]
        return rows

    @staticmethod
    def _add_target(targets: dict, rejected: list, ticker: str, weight: float,
                    prices: dict[str, float], capital: float, *, is_hedge: bool,
                    sleeve_contributions: dict) -> None:
        price = prices.get(ticker)
        if price is None or price <= 0:
            if ticker not in {r["ticker"] for r in rejected}:
                rejected.append({"ticker": ticker, "reason": "no price for sizing"})
            return
        prev = targets.get(ticker)                       # merge name + hedge on same ticker (2a)
        total_w = weight + (prev["weight"] if prev else 0.0)
        targets[ticker] = {
            "weight": total_w,
            "lots": int(round((total_w * capital) / price)),   # lot size 1 in the mock
            "is_hedge": prev["is_hedge"] if prev else is_hedge,
            "sleeve_contributions": {**(prev["sleeve_contributions"] if prev else {}),
                                     **(sleeve_contributions or {})},
        }
