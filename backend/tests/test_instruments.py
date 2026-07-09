"""Instrument-metadata tests: lot rounding, price snapping, FIGI presence + verify gate."""

import json

import pytest

from backend import instruments
from backend.universe import SHARES


def _write_config(tmp_path, all_verified: bool, per_name: dict[str, bool]):
    payload = {
        "schema_version": 1,
        "all_figis_verified": all_verified,
        "instruments": {
            tk: {"ticker": tk, "figi": f"BBG{tk}", "figi_verified": v,
                 "lot": 1, "min_price_step": 0.01, "decimals": 2, "isin": f"RU{tk}"}
            for tk, v in per_name.items()
        },
    }
    p = tmp_path / "instruments.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def test_all_universe_present():
    insts = instruments.load_instruments()
    for tk in SHARES:
        assert tk in insts, tk
        assert insts[tk]["figi"], f"{tk} missing FIGI"
        assert insts[tk]["lot"] >= 1


def test_round_to_lot_floors_to_whole_lots():
    # GAZP lot = 10 -> 23 shares floors to 20
    assert instruments.round_to_lot("GAZP", 23) == 20
    assert instruments.round_to_lot("GAZP", 9) == 0
    # SNGS lot = 100
    assert instruments.round_to_lot("SNGS", 250) == 200
    # SBER lot = 1 -> unchanged
    assert instruments.round_to_lot("SBER", 7) == 7


def test_round_price_snaps_to_min_step():
    # SBER step 0.01
    assert instruments.round_price("SBER", 315.437) == 315.44
    # LKOH step 0.5
    assert instruments.round_price("LKOH", 7001.3) == 7001.5
    # SNGS step 0.005
    assert instruments.round_price("SNGS", 23.4163) == 23.415


def test_figi_for_known_and_unknown():
    assert instruments.figi_for("SBER").startswith("BBG")
    with pytest.raises(KeyError):
        instruments.figi_for("ZZZZ")


def test_committed_config_is_fully_verified():
    """Post T-Invest reconciliation, the shipped config reflects reality: all verified."""
    assert instruments.all_verified() is True
    assert instruments.unverified_figis() == []
    for tk in SHARES:
        assert instruments.get_instrument(tk)["figi_verified"] is True


def test_verify_gate_reports_unverified_names(tmp_path):
    """Gate is False and lists the offender when any single name is unverified."""
    cfg = _write_config(tmp_path, all_verified=False,
                        per_name={"SBER": True, "GAZP": False, "LKOH": True})
    assert instruments.all_verified(cfg) is False
    assert instruments.unverified_figis(cfg) == ["GAZP"]


def test_verify_gate_true_only_when_all_verified(tmp_path):
    cfg = _write_config(tmp_path, all_verified=True,
                        per_name={"SBER": True, "GAZP": True})
    assert instruments.all_verified(cfg) is True
    assert instruments.unverified_figis(cfg) == []
