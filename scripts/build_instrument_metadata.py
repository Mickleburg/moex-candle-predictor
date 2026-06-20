"""Build instrument metadata (FIGI + round-lot + price step) for the 16-name universe.

Execution (T-Invest API) needs a FIGI and a correct round-lot per instrument; it currently
ships placeholder defaults. This emits ``config/instruments.json`` -- a small, shared,
versioned artifact that ``backend.instruments`` exposes to execution and agent.

Two-source design (see docs/DATA_SOURCES.md):
* **MOEX ISS** (authoritative, no auth) -> ``lot``, ``min_price_step``, ``decimals``,
  ``isin``, ``short_name``, ``currency``. Fetched live so e.g. VTBR's post-reverse-split
  lot is correct.
* **T-Invest FIGI** -> curated table below (FIGI is a T-Invest/Bloomberg identifier, not
  published by ISS). Marked ``figi_verified=false``: VALIDATE against a live T-Invest
  instruments dump (sandbox token) before enabling live trading. ``--tinvest-dump FILE``
  cross-checks/overrides FIGIs from a T-Invest instruments JSON export.

Idempotent: re-running on unchanged upstream rewrites identical JSON.

Usage::

    python scripts/build_instrument_metadata.py
    python scripts/build_instrument_metadata.py --tinvest-dump tinvest_shares.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = REPO_ROOT / "config" / "instruments.json"
MOEX_ISS_BASE = "https://iss.moex.com"

UNIVERSE = ("SBER", "GAZP", "LKOH", "GMKN", "ROSN", "NVTK", "TATN", "MGNT",
            "MTSS", "SNGS", "CHMF", "ALRS", "VTBR", "MAGN", "NLMK", "PLZL")

# Curated T-Invest FIGIs (public, stable). Treated as UNVERIFIED until checked against a
# live T-Invest instruments dump -- a wrong FIGI routes an order to the wrong instrument,
# so validation is a hard gate before live (execution is paper-first / is_production=false).
CURATED_FIGI = {
    "SBER": "BBG004730N88", "GAZP": "BBG004730RP0", "LKOH": "BBG004731032",
    "GMKN": "BBG004731489", "ROSN": "BBG004731354", "NVTK": "BBG00475KKY8",
    "TATN": "BBG004RVFFC0", "MGNT": "BBG004RVFCY3", "MTSS": "BBG004S681W1",
    "SNGS": "BBG0047315D0", "CHMF": "BBG00475K6C3", "ALRS": "BBG004S68B31",
    "VTBR": "BBG004730ZJ9", "MAGN": "BBG004S68507", "NLMK": "BBG004S681B4",
    "PLZL": "BBG000R607Y3",
}


def fetch_iss_meta(session: requests.Session, ticker: str) -> dict:
    url = (f"{MOEX_ISS_BASE}/iss/engines/stock/markets/shares/boards/TQBR"
           f"/securities/{ticker}.json")
    resp = session.get(url, params={"iss.meta": "off", "iss.only": "securities"}, timeout=20)
    resp.raise_for_status()
    block = resp.json()["securities"]
    cols = {c: i for i, c in enumerate(block["columns"])}
    if not block["data"]:
        raise ValueError(f"No TQBR securities row for {ticker}")
    row = block["data"][0]

    def g(col):
        return row[cols[col]] if col in cols else None

    return {
        "lot": int(g("LOTSIZE")),
        "min_price_step": float(g("MINSTEP")),
        "decimals": int(g("DECIMALS")),
        "isin": g("ISIN"),
        "short_name": g("LATNAME") or g("SHORTNAME"),
        "currency": (g("CURRENCYID") or "SUR").replace("SUR", "RUB"),
        "board": g("BOARDID"),
        "sectype": g("SECTYPE"),
    }


def load_tinvest_figis(path: Path) -> dict[str, str]:
    """Map ticker -> figi from a T-Invest instruments JSON dump (best-effort schema)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data.get("instruments", data) if isinstance(data, dict) else data
    out: dict[str, str] = {}
    for it in items:
        tk = (it.get("ticker") or "").upper()
        figi = it.get("figi")
        if tk in UNIVERSE and figi:
            out[tk] = figi
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tinvest-dump", default=None,
                    help="T-Invest instruments JSON export to validate/override FIGIs")
    args = ap.parse_args(argv)

    tinvest = load_tinvest_figis(Path(args.tinvest_dump)) if args.tinvest_dump else {}

    session = requests.Session()
    session.headers["User-Agent"] = "moex-instrument-metadata/0.1"

    instruments: dict[str, dict] = {}
    for tk in UNIVERSE:
        meta = fetch_iss_meta(session, tk)
        figi = tinvest.get(tk, CURATED_FIGI.get(tk))
        verified = tk in tinvest
        instruments[tk] = {
            "ticker": tk,
            "figi": figi,
            "figi_verified": verified,
            "figi_source": "t-invest-dump" if verified else "curated",
            **meta,
        }
        flag = "verified" if verified else "CURATED (verify before live)"
        print(f"  {tk:>6}  figi={figi}  lot={meta['lot']:>4}  step={meta['min_price_step']}  "
              f"isin={meta['isin']}  [{flag}]")

    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "moex-iss (lot/step/isin) + t-invest (figi)",
        "all_figis_verified": all(v["figi_verified"] for v in instruments.values()),
        "instruments": instruments,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {len(instruments)} instruments -> {OUT_PATH}")
    print(f"all_figis_verified = {payload['all_figis_verified']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
