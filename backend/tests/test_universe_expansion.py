"""H9 universe-expansion wiring: ADTV passers added (non-required, routed); failers absent."""

from backend import universe as u


def test_core_universe_unchanged_16():
    assert len(u.SHARES) == 16            # FIGI-mapped trading universe stays the 16


def test_expansion_passers_present_nonrequired_routed():
    for tk in ("SBERP", "SNGSP", "PHOR", "MOEX"):
        ins = u.by_key(tk, "1H")
        assert ins is not None, tk
        assert ins.kind == "share"
        assert ins.required is False              # provisional until ML promotes -> WARN not HALT
        assert (ins.engine, ins.market, ins.board) == ("stock", "shares", "TQBR")


def test_adtv_failers_excluded():
    keys = {(i.ticker, i.timeframe) for i in u.INGEST_INSTRUMENTS}
    for tk in ("SIBN", "TATNP", "RTKMP", "BSPB"):   # failed >=300M ADTV screen
        assert (tk, "1H") not in keys, tk


def test_expansion_in_candle_instruments():
    tickers = {i.ticker for i in u.CANDLE_INSTRUMENTS}
    assert {"SBERP", "SNGSP", "PHOR", "MOEX"} <= tickers
