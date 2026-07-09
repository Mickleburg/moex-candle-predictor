"""Offline tests for the T-Invest reconciliation in scripts/build_instrument_metadata.py.

The crux of FIGI verification: a name flips verified ONLY on a full match; any
FIGI/lot/ISIN mismatch is reported as a discrepancy and is NOT silently auto-fixed.
"""

import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import build_instrument_metadata as bim  # noqa: E402


def _inst(tk, figi, lot=1, isin="RU000TEST"):
    return {tk: {"ticker": tk, "figi": figi, "figi_verified": False,
                 "figi_source": "curated", "lot": lot, "isin": isin,
                 "min_price_step": 0.01, "decimals": 2}}


def test_full_match_marks_verified():
    inst = _inst("SBER", "BBG004730N88", lot=1, isin="RU0009029540")
    ti = {"SBER": {"figi": "BBG004730N88", "lot": 1, "isin": "RU0009029540"}}
    disc = bim.reconcile_with_tinvest(inst, ti)
    assert disc == []
    assert inst["SBER"]["figi_verified"] is True
    assert inst["SBER"]["figi_source"] == "t-invest"


def test_figi_mismatch_reported_not_autofixed():
    inst = _inst("SBER", "BBG_CURATED_WRONG")
    ti = {"SBER": {"figi": "BBG_TINVEST_REAL", "lot": 1, "isin": "RU000TEST"}}
    disc = bim.reconcile_with_tinvest(inst, ti)
    assert len(disc) == 1 and disc[0][1] == "mismatch"
    assert "FIGI" in disc[0][2]
    # curated FIGI must NOT be silently overwritten, and name stays unverified
    assert inst["SBER"]["figi"] == "BBG_CURATED_WRONG"
    assert inst["SBER"]["figi_verified"] is False


def test_lot_mismatch_reported():
    inst = _inst("GAZP", "BBG004730RP0", lot=10)
    ti = {"GAZP": {"figi": "BBG004730RP0", "lot": 1, "isin": "RU000TEST"}}
    disc = bim.reconcile_with_tinvest(inst, ti)
    assert len(disc) == 1 and "lot" in disc[0][2]
    assert inst["GAZP"]["figi_verified"] is False


def test_missing_in_tinvest_reported():
    inst = _inst("XXXX", "BBG_X")
    disc = bim.reconcile_with_tinvest(inst, {})
    assert len(disc) == 1 and disc[0][1] == "missing"
    assert inst["XXXX"]["figi_verified"] is False


def test_normalise_prefers_tqbr_board():
    items = [
        {"ticker": "SBER", "classCode": "SPBXM", "figi": "WRONG", "lot": 1, "isin": "X"},
        {"ticker": "SBER", "classCode": "TQBR", "figi": "RIGHT", "lot": 1, "isin": "Y"},
    ]
    out = bim._normalise_tinvest_items(items)
    assert out["SBER"]["figi"] == "RIGHT"
