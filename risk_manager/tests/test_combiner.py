"""Unit tests for the risk_manager combiner.

Covers: netting, the shadow gate (invariants #9/#4: live iff production + forward gate MET),
vol-target x regime gate, limits, sector/market hedge, per-sleeve correlation cap, contract render.

Pure-dict inputs (no ml/ or data/ dependency) so the suite is fast and robust. One optional test
exercises the LIVE 2022 shock case via the ML block when its data/deps are present, else skips.

NOTE: `_sleeve(..., is_production=True)` by default so the risk-layer MATH is exercised on the LIVE
book; the shadow-gate tests pass is_production=False (or a forward-P&L status) explicitly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from risk_manager.src import (  # noqa: E402
    CombinerConfig, combine, shadow_gate_status, to_risk_decisions,
)

CONTRACTS = REPO_ROOT / "contracts"


def _sleeve(positions, sleeve="s3_event", strategy="dividend_runup", market_neutral=True,
            is_production=True):
    return {
        "sleeve": sleeve, "strategy": strategy, "as_of": "2025-06-02 00:00:00+03:00",
        "market_neutral": market_neutral,
        "positions": positions,
        "gross": sum(abs(p["weight"]) for p in positions),
        "model_version": "test_v0", "is_production": is_production,
    }


def _risk_analytics(tickers, vol=0.01, exposure_scalar=1.0, novel=False):
    return {
        "as_of": "2025-06-02T00:00:00+03:00", "timeframe": "1D",
        "horizon": {"bars": 10, "timeframe": "1D"}, "universe": tickers,
        "per_ticker": [{"ticker": t, "vol_forecast": vol, "inv_vol_weight": 1.0 / len(tickers),
                        "valid": True} for t in tickers],
        "regime": {"distance": 1.0, "percentile": 0.5, "novel": novel,
                   "exposure_scalar": exposure_scalar},
        "model_version": "risk_analytics_v0", "is_production": False,
    }


# ── netting ───────────────────────────────────────────────────────────────────────────────────
def test_nets_same_ticker_across_sleeves():
    s1 = _sleeve([{"ticker": "SBER", "weight": 0.2, "leg": "long"}], sleeve="s3_event")
    s2 = _sleeve([{"ticker": "SBER", "weight": 0.1, "leg": "long"},
                  {"ticker": "GAZP", "weight": -0.15, "leg": "short"}], sleeve="s2_macro")
    book = combine([s1, s2], _risk_analytics(["SBER", "GAZP"]), CombinerConfig(hedge_mode="none"))
    by = {p["ticker"]: p for p in book.net_positions}
    assert by["SBER"]["side"] == "LONG"
    assert set(by["SBER"]["sleeve_contributions"]) == {"s3_event", "s2_macro"}
    assert by["GAZP"]["side"] == "SHORT"


def test_opposite_legs_cancel():
    s1 = _sleeve([{"ticker": "SBER", "weight": 0.3, "leg": "long"}])
    s2 = _sleeve([{"ticker": "SBER", "weight": -0.3, "leg": "short"}], sleeve="s1_pairs")
    book = combine([s1, s2], _risk_analytics(["SBER"]), CombinerConfig(hedge_mode="none"))
    assert book.net_positions == []  # fully netted out


# ── shadow gate (invariants #9 / #4) ──────────────────────────────────────────────────────────
def test_nonproduction_sleeve_is_shadow_zero_live():
    """H9 today: is_production=false -> SHADOW book, ZERO live capital, no risk_decisions."""
    book = combine([_sleeve([{"ticker": "SBER", "weight": 0.3, "leg": "long"}],
                            is_production=False)], _risk_analytics(["SBER"]))
    assert book.net_positions == []                       # no live capital
    assert book.risk_scalars["directional_gross"] == 0.0
    assert book.risk_scalars["total_gross"] == 0.0
    assert [p["ticker"] for p in book.shadow_positions] == ["SBER"]   # paper-tracked
    assert book.risk_scalars["shadow_gross"] > 0
    assert book.gating[0]["capital_state"] == "shadow"
    assert book.gating[0]["gate"] == "NOT_MET"
    assert book.is_production is False
    assert to_risk_decisions(book) == []                 # nothing for execution to place


def test_production_sleeve_is_live():
    book = combine([_sleeve([{"ticker": "SBER", "weight": 0.3, "leg": "long"}],
                            is_production=True)], _risk_analytics(["SBER"]))
    assert [p["ticker"] for p in book.net_positions] == ["SBER"]
    assert book.shadow_positions == []
    assert book.gating[0]["capital_state"] == "live"
    assert book.gating[0]["gate"] == "MET"


def test_force_live_disables_gate():
    cfg = CombinerConfig(require_production_for_live=False)
    book = combine([_sleeve([{"ticker": "SBER", "weight": 0.3, "leg": "long"}],
                            is_production=False)], _risk_analytics(["SBER"]), cfg)
    assert [p["ticker"] for p in book.net_positions] == ["SBER"]   # gate disabled -> live
    assert book.gating[0]["gate"] == "DISABLED"


def test_forward_pnl_gate_forces_shadow_even_if_production():
    """A signed-off sleeve with a NEGATIVE forward-P&L attribution stays shadow (gate NOT MET)."""
    pos = [{"ticker": "SBER", "weight": 0.3, "leg": "long"}]
    status = {"s3_event": {"gate": "NOT_MET", "reason": "forward n=12 net -0.93%"}}
    book = combine([_sleeve(pos, is_production=True)], _risk_analytics(["SBER"]),
                   sleeve_status=status)
    assert book.net_positions == []
    assert book.gating[0]["capital_state"] == "shadow"
    # also via a raw forward_pnl number
    book2 = combine([_sleeve(pos, is_production=True)], _risk_analytics(["SBER"]),
                    sleeve_status={"s3_event": {"forward_pnl": -0.0093}})
    assert book2.net_positions == []


def test_shadow_gate_status_helper():
    sig = _sleeve([{"ticker": "SBER", "weight": 0.3, "leg": "long"}], is_production=False)
    g = shadow_gate_status(sig)
    assert g["capital_state"] == "shadow" and g["gate"] == "NOT_MET"
    g2 = shadow_gate_status({**sig, "is_production": True})
    assert g2["capital_state"] == "live" and g2["gate"] == "MET"


def test_book_is_production_reflects_live_sleeves():
    pos = [{"ticker": "SBER", "weight": 0.3, "leg": "long"}]
    assert combine([_sleeve(pos, is_production=False)], _risk_analytics(["SBER"])).is_production is False
    assert combine([_sleeve(pos, is_production=True)], _risk_analytics(["SBER"])).is_production is True
    # a non-production sleeve mixed in is shadow-gated, so it cannot taint the live book's flag
    mixed = combine([_sleeve(pos, is_production=True),
                     _sleeve([{"ticker": "GAZP", "weight": 0.2, "leg": "long"}],
                             sleeve="s2_macro", is_production=False)],
                    _risk_analytics(["SBER", "GAZP"]))
    assert mixed.is_production is True
    assert {p["ticker"] for p in mixed.net_positions} == {"SBER"}          # only the live sleeve
    assert {p["ticker"] for p in mixed.shadow_positions} == {"GAZP"}       # shadow sleeve paper-only


# ── regime gate (H5) on the live book ─────────────────────────────────────────────────────────
def test_regime_gate_cuts_gross_when_novel():
    positions = [{"ticker": "SBER", "weight": 0.2, "leg": "long"},
                 {"ticker": "GAZP", "weight": 0.2, "leg": "long"}]
    tickers = ["SBER", "GAZP"]
    cfg = CombinerConfig(max_vol_leverage=1.0, target_book_vol_annual=100.0)  # pin vol_scalar=1.0
    normal = combine([_sleeve(positions)], _risk_analytics(tickers, exposure_scalar=1.0, novel=False), cfg)
    gated = combine([_sleeve(positions)], _risk_analytics(tickers, exposure_scalar=0.2, novel=True), cfg)
    assert gated.risk_scalars["exposure_scalar"] == pytest.approx(0.2)
    assert gated.risk_scalars["regime_novel"] is True
    assert gated.risk_scalars["directional_gross"] == pytest.approx(
        0.2 * normal.risk_scalars["directional_gross"], rel=1e-6)
    assert "regime_gate" in gated.limits["binding"]


def test_regime_gate_zero_exposure_empties_book():
    book = combine([_sleeve([{"ticker": "SBER", "weight": 0.3, "leg": "long"}])],
                   _risk_analytics(["SBER"], exposure_scalar=0.0, novel=True), CombinerConfig())
    assert book.net_positions == []
    assert book.risk_scalars["total_gross"] == 0.0


# ── limits ────────────────────────────────────────────────────────────────────────────────────
def test_all_limits_respected():
    positions = [{"ticker": "LKOH", "weight": 0.36, "leg": "long"},
                 {"ticker": "TATN", "weight": 0.30, "leg": "long"},   # LKOH+TATN = MOEXOG
                 {"ticker": "SBER", "weight": 0.34, "leg": "long"}]
    cfg = CombinerConfig(max_name_weight=0.34, max_sector_gross=0.60, max_gross=1.0)
    book = combine([_sleeve(positions)], _risk_analytics(["LKOH", "TATN", "SBER"]), cfg)
    assert book.limits["name_caps_ok"] and book.limits["sector_caps_ok"] and book.limits["gross_cap_ok"]
    for p in book.net_positions:
        assert abs(p["weight"]) <= cfg.max_name_weight + 1e-6
    sec = {}
    for p in book.net_positions:
        sec[p["sector"]] = sec.get(p["sector"], 0.0) + abs(p["weight"])
    assert all(g <= cfg.max_sector_gross + 1e-6 for g in sec.values())
    assert book.risk_scalars["directional_gross"] <= cfg.max_gross + 1e-6
    assert "sector_cap:MOEXOG" in book.limits["binding"]


def test_name_cap_clips_even_with_vol_leverage():
    positions = [{"ticker": "SBER", "weight": 0.5, "leg": "long"}]
    cfg = CombinerConfig(max_name_weight=0.34, target_book_vol_annual=100.0, max_vol_leverage=1.5)
    book = combine([_sleeve(positions)], _risk_analytics(["SBER"], vol=0.001), cfg)
    assert book.net_positions[0]["weight"] == pytest.approx(0.34)


def test_per_sleeve_gross_cap_correlation_cap():
    """max_sleeve_gross caps any single sleeve's directional gross (correlation cap by sleeve-id)."""
    big = _sleeve([{"ticker": "SBER", "weight": 0.5, "leg": "long"},
                   {"ticker": "MTSS", "weight": 0.5, "leg": "long"}])   # sleeve gross 1.0
    cfg = CombinerConfig(max_sleeve_gross=0.5, max_vol_leverage=1.0, target_book_vol_annual=100.0,
                         hedge_mode="none")
    book = combine([big], _risk_analytics(["SBER", "MTSS"]), cfg)
    # the sleeve's gross was scaled to 0.5 before netting -> each name 0.25
    assert book.risk_scalars["directional_gross"] == pytest.approx(0.5, abs=1e-6)
    assert "sleeve_cap:s3_event" in book.limits["binding"]


# ── hedge ─────────────────────────────────────────────────────────────────────────────────────
def test_sector_hedge_neutralizes_each_sector():
    positions = [{"ticker": "LKOH", "weight": 0.2, "leg": "long"},   # MOEXOG
                 {"ticker": "TATN", "weight": 0.2, "leg": "long"},   # MOEXOG
                 {"ticker": "SBER", "weight": 0.2, "leg": "long"}]   # MOEXFN
    cfg = CombinerConfig(hedge_mode="sector", max_vol_leverage=1.0, target_book_vol_annual=100.0)
    book = combine([_sleeve(positions)], _risk_analytics(["LKOH", "TATN", "SBER"]), cfg)
    legs = {leg["instrument"]: leg["weight"] for leg in book.hedge["legs"]}
    assert book.hedge["mode"] == "sector"
    assert legs["MOEXOG"] == pytest.approx(-0.4, abs=1e-6)
    assert legs["MOEXFN"] == pytest.approx(-0.2, abs=1e-6)


def test_market_hedge_single_index_leg():
    positions = [{"ticker": "LKOH", "weight": 0.2, "leg": "long"},
                 {"ticker": "SBER", "weight": 0.2, "leg": "long"}]
    cfg = CombinerConfig(hedge_mode="market", max_vol_leverage=1.0, target_book_vol_annual=100.0)
    book = combine([_sleeve(positions)], _risk_analytics(["LKOH", "SBER"]), cfg)
    assert book.hedge["mode"] == "market"
    assert len(book.hedge["legs"]) == 1
    assert book.hedge["legs"][0]["instrument"] == "IMOEX"
    assert book.hedge["legs"][0]["weight"] == pytest.approx(-0.4, abs=1e-6)


def test_sleeve_suggested_hedge_is_dropped_for_book_hedge():
    positions = [{"ticker": "SBER", "weight": 0.2, "leg": "long"},
                 {"ticker": "IMOEX", "weight": -0.2, "leg": "hedge"}]
    book = combine([_sleeve(positions)], _risk_analytics(["SBER"]),
                   CombinerConfig(hedge_mode="sector", max_vol_leverage=1.0, target_book_vol_annual=100.0))
    tickers = {p["ticker"] for p in book.net_positions}
    assert "IMOEX" not in tickers
    assert book.hedge["legs"][0]["instrument"] == "MOEXFN"


# ── ranking-form adapter (S1/S2 shape) ──────────────────────────────────────────────────────
def test_consumes_ranking_form_aggregated_signal():
    agg = {
        "as_of": "2025-05-15T15:00:00+03:00", "timeframe": "1D", "sleeve": "s2_macro",
        "universe": ["SBER", "LKOH", "GAZP"],
        "rankings": [
            {"ticker": "SBER", "score": 1.4, "rank": 1, "percentile": 0.95, "leg": "long"},
            {"ticker": "LKOH", "score": 0.0, "rank": 2, "percentile": 0.5, "leg": "flat"},
            {"ticker": "GAZP", "score": -1.2, "rank": 3, "percentile": 0.05, "leg": "short"},
        ],
        "market_neutral": True, "model_version": "xsec_v0", "is_production": True,
    }
    book = combine([agg], _risk_analytics(["SBER", "LKOH", "GAZP"]), CombinerConfig(hedge_mode="none"))
    by = {p["ticker"]: p for p in book.net_positions}
    assert by["SBER"]["side"] == "LONG" and by["GAZP"]["side"] == "SHORT"
    assert "LKOH" not in by  # flat leg -> no position


# ── contract render + invariants ──────────────────────────────────────────────────────────────
def test_renders_valid_risk_decisions():
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads((CONTRACTS / "risk_decision.schema.json").read_text(encoding="utf-8"))
    positions = [{"ticker": "LKOH", "weight": 0.3, "leg": "long"},
                 {"ticker": "SBER", "weight": 0.3, "leg": "long"}]
    book = combine([_sleeve(positions)], _risk_analytics(["LKOH", "SBER"]), CombinerConfig())
    decisions = to_risk_decisions(book)
    assert len(decisions) >= 1
    for d in decisions:
        jsonschema.Draft202012Validator(schema).validate(d)


def test_risk_book_validates_against_schema():
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads((CONTRACTS / "risk_book.schema.json").read_text(encoding="utf-8"))
    positions = [{"ticker": "LKOH", "weight": 0.3, "leg": "long"},
                 {"ticker": "SBER", "weight": 0.3, "leg": "long"}]
    # validate both a live book and a shadow (gated) book
    for prod in (True, False):
        book = combine([_sleeve(positions, is_production=prod)], _risk_analytics(["LKOH", "SBER"]),
                       CombinerConfig())
        jsonschema.Draft202012Validator(schema).validate(book.to_dict())


def test_empty_sleeve_yields_empty_book():
    book = combine([_sleeve([], is_production=False)], _risk_analytics(["SBER"]))
    assert book.net_positions == []
    assert book.shadow_positions == []
    assert book.hedge["legs"] == []
    assert book.risk_scalars["total_gross"] == 0.0
    assert book.is_production is False


# ── optional: LIVE 2022 shock case via the ML block (skips without data/deps) ────────────────
def test_live_2022_shock_cuts_gross():
    pytest.importorskip("pandas")
    sys.path.insert(0, str(REPO_ROOT / "ml"))
    try:
        import pandas as pd
        from src.features.cross_sectional import load_panels
        from src.service.dividend_sleeve import build_sleeve_signal, load_dividend_calendar
        from src.service.risk_analytics import build_risk_analytics
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"ML block/data unavailable: {exc}")

    uni = ["SBER", "GAZP", "LKOH", "GMKN", "ROSN", "NVTK", "TATN", "MGNT",
           "MTSS", "SNGS", "CHMF", "ALRS"]
    try:
        panel, _, market = load_panels(uni, "1D")
        cal = load_dividend_calendar()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"ML data files unavailable: {exc}")

    as_of = pd.Timestamp("2022-04-15", tz="Europe/Moscow")
    sig = build_sleeve_signal(panel, cal, as_of)
    ra = build_risk_analytics(panel, market, as_of=as_of)
    if not any(p["leg"] == "long" for p in sig["positions"]):
        pytest.skip("no active run-up names on the probed 2022 date")
    assert ra["regime"]["novel"] is True
    # force-live to observe the regime gate cutting the LIVE book (H9 is is_production=false -> shadow)
    cfg = CombinerConfig(require_production_for_live=False)
    gated = combine([sig], ra, cfg)
    ungated = combine([sig], {**ra, "regime": {**ra["regime"], "exposure_scalar": 1.0}}, cfg)
    assert gated.risk_scalars["directional_gross"] < ungated.risk_scalars["directional_gross"]
