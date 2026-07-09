"""Sector membership + hedge instruments for the risk_manager combiner.

Mirrors the ML universe's ticker -> MOEX sector-index mapping (ml/src/features/cross_sectional.py
SECTOR_MAP). Kept as a local copy on purpose: risk_manager must not import from ml/. This is static
reference data (MOEX sector membership), extended to the 16-name H9 sleeve universe.

The sector index doubles as the per-name SECTOR HEDGE instrument. P0 analysis (H9) found the
sector hedge dominant for the dividend run-up (Sharpe +0.92 / DD -0.105) vs a market beta=1 IMOEX
hedge (+0.54 / -0.173) — the run-up is a name-vs-sector effect, so we hedge the sector, not the index.
"""

from __future__ import annotations

MARKET_INDEX = "IMOEX"

# ticker -> MOEX sector index. Mirror of ml SECTOR_MAP, extended for the H9 sleeve universe.
SECTOR_MAP: dict[str, str] = {
    # Financials
    "SBER": "MOEXFN", "VTBR": "MOEXFN",
    # Oil & gas
    "GAZP": "MOEXOG", "LKOH": "MOEXOG", "ROSN": "MOEXOG",
    "NVTK": "MOEXOG", "TATN": "MOEXOG", "SNGS": "MOEXOG",
    # Metals & mining
    "GMKN": "MOEXMM", "CHMF": "MOEXMM", "ALRS": "MOEXMM",
    "MAGN": "MOEXMM", "NLMK": "MOEXMM", "PLZL": "MOEXMM",
    # Consumer
    "MGNT": "MOEXCN",
    # Telecom
    "MTSS": "MOEXTL",
}

# Index tickers that may appear on a sleeve's suggested hedge leg (so the combiner can tell a
# directional name from an index hedge it is allowed to re-derive).
INDEX_TICKERS: frozenset[str] = frozenset(
    {MARKET_INDEX, "RTSI"} | set(SECTOR_MAP.values())
)


def sector_of(ticker: str) -> str:
    """Sector index for a ticker; falls back to the market index for unmapped names."""
    return SECTOR_MAP.get(ticker, MARKET_INDEX)


def is_index(ticker: str) -> bool:
    """True if `ticker` is a market/sector index (a hedge instrument), not a tradable name."""
    return ticker in INDEX_TICKERS
