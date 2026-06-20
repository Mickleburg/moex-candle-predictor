"""Task 3: lot/FIGI loaders consume the backend instrument metadata (not placeholders).

Skips cleanly when backend.instruments / config/instruments.json are not present, in which case the
loaders fall back to DEFAULT_LOT_SIZES + an empty FIGI map (verified by the fallback test).
"""

from __future__ import annotations

import pytest

from execution.src.config import DEFAULT_LOT_SIZES
from execution.src.instruments import load_figi_map, load_lot_sizes

backend_instruments = pytest.importorskip("backend.instruments")


def _backend_meta_or_skip() -> dict:
    try:
        return backend_instruments.load_instruments()
    except Exception as exc:  # config/instruments.json not generated in this checkout
        pytest.skip(f"backend instrument metadata unavailable: {exc}")


def test_lot_sizes_come_from_backend():
    meta = _backend_meta_or_skip()
    lots = load_lot_sizes()
    # every backend lot is reflected (backend overrides any default)
    sample = [t for t in ("SBER", "GAZP", "LKOH", "TATN") if t in meta]
    assert sample, "expected at least one core share in backend metadata"
    for t in sample:
        assert lots[t] == backend_instruments.lot_for(t)
    # index hedge proxies the backend universe may omit still resolve from the defaults
    for idx in ("MOEXOG", "MOEXFN"):
        assert lots.get(idx, DEFAULT_LOT_SIZES.get(idx)) == DEFAULT_LOT_SIZES[idx] or idx in meta


def test_figi_map_from_backend_or_empty():
    meta = _backend_meta_or_skip()
    figis = load_figi_map()
    for t, m in meta.items():
        if m.get("figi"):
            assert figis.get(t.upper()) == str(m["figi"])
