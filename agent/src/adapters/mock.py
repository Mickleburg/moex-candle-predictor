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
                 model_version: str = "h9_dividend_runup_v1_mock"):
        # default: 3 names in their pre-ex window, capped inverse-vol-ish weights summing to ~1.
        self._longs = longs if longs is not None else {"SBER": 0.34, "LKOH": 0.34, "TATN": 0.32}
        self._abstain = abstain
        self._model_version = model_version

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
            "is_production": False,
        }


class MockCombiner:
    """Stands in for the risk_manager combiner. Self-contained netting + the risk layer knobs
    the orchestrator reacts to (exposure_scalar from the H5 regime gate, vol scalar, caps)."""

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

    def combine(self, sleeve_signals: list[dict], as_of: str) -> dict:
        # 1. net directional weights by ticker; track per-sleeve contributions
        net: dict[str, float] = {}
        contrib: dict[str, dict[str, float]] = {}
        for sig in sleeve_signals:
            for p in sig.get("positions", []):
                if p.get("leg") not in ("long", "short"):
                    continue
                w = float(p["weight"])
                net[p["ticker"]] = net.get(p["ticker"], 0.0) + w
                contrib.setdefault(p["ticker"], {})[sig["sleeve"]] = \
                    contrib.get(p["ticker"], {}).get(sig["sleeve"], 0.0) + w

        binding: list[str] = []
        # 2. name cap
        capped = {}
        for t, w in net.items():
            if abs(w) > self._max_name + 1e-9:
                binding.append(f"name_cap:{t}")
                capped[t] = self._max_name if w > 0 else -self._max_name
            else:
                capped[t] = w
        # 3. risk scalars: vol-target x regime gate
        book_scalar = self._vol_scalar * self._exposure_scalar
        if abs(self._vol_scalar - 1.0) > 1e-9:
            binding.append("vol_target")
        if self._exposure_scalar < 1.0 - 1e-9:
            binding.append("regime_gate")
        scaled = {t: w * book_scalar for t, w in capped.items()}
        # 4. gross cap (names only)
        gross = sum(abs(w) for w in scaled.values())
        if gross > self._max_gross + 1e-9:
            binding.append("gross_cap")
            s = self._max_gross / gross
            scaled = {t: w * s for t, w in scaled.items()}
        scaled = {t: w for t, w in scaled.items() if abs(w) > 1e-9}
        directional_gross = sum(abs(w) for w in scaled.values())

        net_positions = [
            {"ticker": t, "weight": round(w, 6), "side": "LONG" if w > 0 else "SHORT",
             "sector": _SECTOR.get(t, "IMOEX"),
             "sleeve_contributions": {k: round(v, 6) for k, v in contrib.get(t, {}).items()}}
            for t, w in sorted(scaled.items(), key=lambda kv: -abs(kv[1]))
        ]
        hedge = self._build_hedge(scaled)
        hedge_gross = sum(abs(leg["weight"]) for leg in hedge["legs"])

        sec_gross: dict[str, float] = {}
        for p in net_positions:
            sec_gross[p["sector"]] = sec_gross.get(p["sector"], 0.0) + abs(p["weight"])

        return {
            "as_of": as_of,
            "timeframe": self._timeframe,
            "sleeves": [{"sleeve": s["sleeve"], "strategy": s.get("strategy", ""),
                         "gross": round(sum(abs(float(p["weight"])) for p in s.get("positions", [])), 6)}
                        for s in sleeve_signals],
            "net_positions": net_positions,
            "hedge": hedge,
            "risk_scalars": {
                "target_book_vol_annual": 0.12,
                "book_vol_estimate_annual": 0.0,
                "vol_scalar": round(self._vol_scalar, 6),
                "exposure_scalar": round(self._exposure_scalar, 6),
                "regime_novel": bool(self._regime_novel),
                "directional_gross": round(directional_gross, 6),
                "total_gross": round(directional_gross + hedge_gross, 6),
            },
            "limits": {
                "max_name_weight": self._max_name,
                "max_sector_gross": 0.6,
                "max_gross": self._max_gross,
                "name_caps_ok": True,
                "sector_caps_ok": True,
                "gross_cap_ok": directional_gross <= self._max_gross + 1e-6,
                "binding": sorted(set(binding)),
            },
            "model_version": "risk_combiner_v0_mock",
            "is_production": False,
        }

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
    """Stands in for the execution block: a deterministic paper broker.

    Reconciles the target risk_book against the current book, emits LIMIT delta-orders with
    stable client_order_ids (dedup), and fills them at the reference price. Lot rounding uses
    lot size 1 for the mock. The −12/−2 entry/exit DISCIPLINE is already baked into which
    names the sleeve emits, so daily target-book reconciliation here enters new names and
    closes names that left their window. dry-run prints orders without changing the book.
    """

    def reconcile_and_execute(self, *, risk_book: dict, positions: list[dict],
                              prices: dict[str, float], capital: float, mode: str,
                              trade_date: str, phase: str) -> ExecutionResult:
        if mode == "live":
            raise RuntimeError("PaperBrokerExecution cannot run in live mode (mock broker)")

        # build target lots from net_positions + hedge legs
        targets: dict[str, dict] = {}
        rejected: list[dict] = []
        for p in risk_book.get("net_positions", []):
            self._add_target(targets, rejected, p["ticker"], float(p["weight"]), prices, capital,
                             is_hedge=False, sleeve_contributions=p.get("sleeve_contributions", {}))
        for leg in risk_book.get("hedge", {}).get("legs", []):
            self._add_target(targets, rejected, leg["instrument"], float(leg["weight"]), prices,
                             capital, is_hedge=True, sleeve_contributions={})

        current = {p["ticker"]: int(p["lots"]) for p in positions}
        orders, reports = [], []
        new_positions: dict[str, dict] = {}

        for ticker in sorted(set(targets) | set(current)):
            target_lots = targets.get(ticker, {}).get("lots", 0)
            cur_lots = current.get(ticker, 0)
            delta = target_lots - cur_lots
            price = prices.get(ticker)
            meta = targets.get(ticker, {})
            if delta != 0 and price is not None:
                side = "BUY" if delta > 0 else "SELL"
                coid = f"{trade_date}-{phase}-{ticker}-{'B' if delta > 0 else 'S'}"
                order = {"ticker": ticker, "side": side, "quantity_lots": abs(delta),
                         "order_type": "LIMIT", "limit_price": round(price, 4), "client_order_id": coid}
                orders.append(order)
                status = "DRY_RUN" if mode == "dry-run" else "FILLED"
                reports.append({"client_order_id": coid, "ticker": ticker, "status": status,
                                "exchange_order_id": None if mode == "dry-run" else f"mock-{coid}",
                                "filled_quantity_lots": 0 if mode == "dry-run" else abs(delta),
                                "avg_fill_price": None if mode == "dry-run" else round(price, 4),
                                "message": f"paper {status.lower()} {side} {abs(delta)} {ticker}"})

            # resulting book: dry-run keeps current; paper moves to target
            final_lots = cur_lots if mode == "dry-run" else target_lots
            if final_lots != 0:
                new_positions[ticker] = {
                    "ticker": ticker, "lots": final_lots,
                    "avg_price": round(price, 4) if price is not None else 0.0,
                    "last_price": round(price, 4) if price is not None else None,
                    "is_hedge": meta.get("is_hedge", False),
                    "sleeve_contributions": meta.get("sleeve_contributions", {}),
                }

        return ExecutionResult(orders=orders, reports=reports,
                               positions=list(new_positions.values()), rejected=rejected)

    @staticmethod
    def _add_target(targets: dict, rejected: list, ticker: str, weight: float,
                    prices: dict[str, float], capital: float, *, is_hedge: bool,
                    sleeve_contributions: dict) -> None:
        price = prices.get(ticker)
        if price is None or price <= 0:
            rejected.append({"ticker": ticker, "reason": "no price for sizing"})
            return
        lots = int(round((weight * capital) / price))   # lot size 1 in the mock
        targets[ticker] = {"lots": lots, "is_hedge": is_hedge,
                           "sleeve_contributions": sleeve_contributions}
