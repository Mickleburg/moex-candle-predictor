"""Instrument universe for the autonomous ingest + integrity gate.

Single source of truth for *what* the EOD job keeps fresh. Fetch parameters (ISS
engine/market/board) are reused from ``scripts/download_candles.py::INSTRUMENT_REGISTRY``
-- this module only declares the universe and per-instrument timeframes, it does not
duplicate endpoint knowledge.

Kinds
-----
* ``share``              -- TQBR equities, the H9 sleeve trades these (1H + 1D).
* ``index``              -- broad-market / sector / rates context (some 1D-only on ISS).
* ``continuous_future``  -- Brent / gas, stitched front-month (rebuilt by
  ``scripts/download_futures_continuous.py``, not a plain candle download).

``REQUIRED_FOR_TRADING`` is the subset whose freshness the integrity gate treats as
HALT-worthy: the names the sleeve longs + the market-context series risk_manager hedges
with. A stale sector/secondary index warns but does not by itself HALT.
"""

from __future__ import annotations

from dataclasses import dataclass

# Original 12-name liquid universe -- maintained at BOTH 1H and 1D (full history).
CORE_SHARES = (
    "SBER", "GAZP", "LKOH", "GMKN", "ROSN", "NVTK", "TATN", "MGNT",
    "MTSS", "SNGS", "CHMF", "ALRS",
)
# H7 pairs-breadth additions -- currently seeded at 1H only (1D pending a backfill).
EXTRA_SHARES = ("VTBR", "MAGN", "NLMK", "PLZL")
SHARES = CORE_SHARES + EXTRA_SHARES  # full 16-name FIGI-mapped trading universe

# H9 universe-expansion lines (ml/docs/research/h9_universe_expansion_2026-06-21.md), seeded at
# 1H from 2020. PASSED the a-priori >=300 M RUB ADTV screen (criterion 1, run 2026-06-24):
#   SBERP 457M, SNGSP 712M, PHOR 363M, MOEX 575M  -> included
# FAILED the screen and EXCLUDED (not seeded): SIBN 241M, TATNP 211M, BSPB 153M, RTKMP 51M.
# Provisional-for-trading: maintained + integrity-checked (required=False -> WARN, not HALT) until
# the ML IS study confirms the run-up edge on these lines and promotes them. Not yet FIGI-mapped
# (config/instruments.json stays the 16); unverified FIGI is acceptable for paper.
EXPANSION_SHARES = ("SBERP", "SNGSP", "PHOR", "MOEX")

# Indices available at 1H on ISS (intraday context).
INDICES_1H = ("IMOEX", "RTSI", "MOEXFN", "MOEXOG", "RGBI")
# Indices we keep at daily resolution (incl. ISS 1D-only sector indices).
INDICES_1D = ("IMOEX", "RTSI", "MOEXFN", "MOEXOG", "MOEXMM", "MOEXCN", "MOEXTL", "RGBI")
# Continuous front-month futures (1H), rebuilt by the futures stitcher.
CONTINUOUS_FUTURES = ("BR_CONT", "NG_CONT")

# Series whose freshness/integrity is HALT-worthy (must exist + be fresh to trade the
# H9 sleeve and its hedge). Everything else is context: a failure WARNs, not HALTs.
# 1D of the EXTRA_SHARES and of RTSI/RGBI is not seeded yet -> left non-required (WARN)
# until a backfill promotes them.
_REQUIRED_INDEX_1H = ("IMOEX", "RGBI")
_REQUIRED_INDEX_1D = ("IMOEX",)


@dataclass(frozen=True)
class Instrument:
    ticker: str
    timeframe: str
    kind: str            # "share" | "index" | "continuous_future"
    required: bool = False  # freshness failure -> HALT (vs warn)
    asset: str = ""      # for continuous_future: BR / NG (stitcher --asset)
    # ISS routing override for names NOT in scripts/download_candles.py::INSTRUMENT_REGISTRY
    # (e.g. the H9 expansion lines). Empty -> ingest falls back to the registry.
    engine: str = ""
    market: str = ""
    board: str = ""


def _build() -> list[Instrument]:
    out: list[Instrument] = []
    for t in SHARES:
        out.append(Instrument(t, "1H", "share", required=True))
        # 1D is required only for the core 12 (the EXTRA_SHARES have no 1D yet).
        out.append(Instrument(t, "1D", "share", required=(t in CORE_SHARES)))
    for t in EXPANSION_SHARES:
        # 1H only, non-required (provisional until ML promotes); explicit TQBR routing.
        out.append(Instrument(t, "1H", "share", required=False,
                              engine="stock", market="shares", board="TQBR"))
    for t in INDICES_1H:
        out.append(Instrument(t, "1H", "index", required=(t in _REQUIRED_INDEX_1H)))
    for t in INDICES_1D:
        out.append(Instrument(t, "1D", "index", required=(t in _REQUIRED_INDEX_1D)))
    for t in CONTINUOUS_FUTURES:
        out.append(Instrument(t, "1H", "continuous_future",
                              required=(t == "BR_CONT"), asset=t.split("_")[0]))
    # de-dup (IMOEX/RGBI appear in both index lists with distinct timeframes already)
    seen, uniq = set(), []
    for ins in out:
        key = (ins.ticker, ins.timeframe)
        if key not in seen:
            seen.add(key)
            uniq.append(ins)
    return uniq


INGEST_INSTRUMENTS: list[Instrument] = _build()

# Convenience views ----------------------------------------------------------
CANDLE_INSTRUMENTS = [i for i in INGEST_INSTRUMENTS if i.kind in ("share", "index")]
FUTURE_INSTRUMENTS = [i for i in INGEST_INSTRUMENTS if i.kind == "continuous_future"]
REQUIRED_FOR_TRADING = [i for i in INGEST_INSTRUMENTS if i.required]


def by_key(ticker: str, timeframe: str) -> Instrument | None:
    for i in INGEST_INSTRUMENTS:
        if i.ticker == ticker and i.timeframe == timeframe:
            return i
    return None
