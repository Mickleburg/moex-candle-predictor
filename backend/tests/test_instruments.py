"""Instrument-metadata tests: lot rounding, price snapping, FIGI presence + verify gate."""

import pytest

from backend import instruments
from backend.universe import SHARES


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


def test_verify_gate_blocks_live_until_validated():
    # curated FIGIs are unverified by construction -> live gate must report False
    assert instruments.all_verified() is False
    assert set(instruments.unverified_figis()) >= set(SHARES)
