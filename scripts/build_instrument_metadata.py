"""Build instrument metadata (FIGI + round-lot + price step) for the 16-name universe.

Execution (T-Invest API) needs a FIGI and a correct round-lot per instrument; it currently
ships placeholder defaults. This emits ``config/instruments.json`` -- a small, shared,
versioned artifact that ``backend.instruments`` exposes to execution and agent.

Two-source design (see docs/DATA_SOURCES.md):
* **MOEX ISS** (authoritative, no auth) -> ``lot``, ``min_price_step``, ``decimals``,
  ``isin``, ``short_name``, ``currency``. Fetched live so e.g. VTBR's post-reverse-split
  lot is correct.
* **T-Invest FIGI** -> curated table below (FIGI is a T-Invest/Bloomberg identifier, not
  published by ISS). Marked ``figi_verified=false`` until reconciled against the real
  T-Invest InstrumentsService.

Verification (``--verify-tinvest``): reads ``TINVEST_TOKEN`` from ``.env`` (sandbox /
read-only; never committed), pulls the T-Invest TQBR share list, and cross-checks
ticker<->FIGI<->lot<->ISIN against each name. A name flips ``figi_verified=true`` ONLY when
all three agree. Discrepancies (FIGI or lot mismatch) are NOT silently auto-fixed -- they
are reported explicitly for a human decision, and the name stays unverified.
``all_figis_verified`` is true only when every one of the 16 names reconciles -> this is the
FIGI gate the execution block checks (live still additionally gated by EXECUTION_ALLOW_LIVE).
``--tinvest-dump FILE`` does the same reconciliation offline from a saved JSON export.

Idempotent: re-running on unchanged upstream rewrites identical JSON (bar ``generated_at``).

Usage::

    python scripts/build_instrument_metadata.py                  # ISS + curated FIGI (unverified)
    python scripts/build_instrument_metadata.py --verify-tinvest  # reconcile vs live T-Invest
    python scripts/build_instrument_metadata.py --tinvest-dump shares.json  # offline reconcile
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = REPO_ROOT / "config" / "instruments.json"
ENV_PATH = REPO_ROOT / ".env"
MOEX_ISS_BASE = "https://iss.moex.com"
TINVEST_REST = ("https://invest-public-api.tinkoff.ru/rest/"
                "tinkoff.public.invest.api.contract.v1.InstrumentsService")

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


def _normalise_tinvest_items(items: list[dict]) -> dict[str, dict]:
    """Index T-Invest share items by ticker, keeping the TQBR main-board entry.

    A ticker can appear on several boards/currencies; we want the MOEX main board
    (classCode TQBR) to match the ISS source.
    """
    out: dict[str, dict] = {}
    for it in items:
        tk = (it.get("ticker") or "").upper()
        if tk not in UNIVERSE:
            continue
        cls = it.get("classCode") or it.get("class_code")
        if out.get(tk) and cls != "TQBR":
            continue  # already have a (preferably TQBR) row
        lot = it.get("lot")
        out[tk] = {
            "figi": it.get("figi"),
            "lot": int(lot) if lot is not None else None,
            "isin": it.get("isin"),
            "class_code": cls,
            "currency": (it.get("currency") or "").upper(),
            "name": it.get("name"),
        }
    return out


def load_tinvest_dump(path: Path) -> dict[str, dict]:
    """Reconciliation map ticker -> {figi, lot, isin, ...} from a saved T-Invest export."""
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data.get("instruments", data) if isinstance(data, dict) else data
    return _normalise_tinvest_items(items)


def read_env_token(var: str = "TINVEST_TOKEN") -> str:
    """Read a secret from the process env or the local (gitignored) .env. Never logged."""
    if os.environ.get(var):
        return os.environ[var]
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            if k.strip() == var:
                return v.strip().strip('"').strip("'")
    raise RuntimeError(f"{var} not found in environment or {ENV_PATH} (read-only token expected)")


def fetch_tinvest_shares(token: str,
                         status: str = "INSTRUMENT_STATUS_ALL") -> dict[str, dict]:
    """Pull the T-Invest share list (InstrumentsService/Shares) and index by ticker.

    Reference data is read-only and works with a sandbox/read-only token. The token is sent
    only in the Authorization header; it is never printed.
    """
    resp = requests.post(
        f"{TINVEST_REST}/Shares",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json={"instrumentStatus": status},
        timeout=30,
    )
    resp.raise_for_status()
    return _normalise_tinvest_items(resp.json().get("instruments", []))


def reconcile_with_tinvest(instruments: dict[str, dict],
                           tinvest: dict[str, dict]) -> list[tuple[str, str, str]]:
    """Cross-check each name vs the T-Invest reference; flip figi_verified only on full match.

    Returns a list of (ticker, kind, detail) discrepancies. Mismatches are NOT auto-fixed:
    the curated FIGI is left in place and the name stays unverified for a human to resolve.
    """
    discrepancies: list[tuple[str, str, str]] = []
    for tk, inst in instruments.items():
        ti = tinvest.get(tk)
        if ti is None:
            inst["figi_verified"] = False
            inst["figi_source"] = "curated (UNVERIFIED: absent in T-Invest TQBR list)"
            discrepancies.append((tk, "missing", "not found in T-Invest TQBR shares"))
            continue
        issues: list[str] = []
        if ti.get("figi") != inst["figi"]:
            issues.append(f"FIGI curated={inst['figi']} t-invest={ti.get('figi')}")
        if ti.get("lot") is not None and ti["lot"] != inst["lot"]:
            issues.append(f"lot iss={inst['lot']} t-invest={ti['lot']}")
        if ti.get("isin") and inst.get("isin") and ti["isin"] != inst["isin"]:
            issues.append(f"ISIN iss={inst['isin']} t-invest={ti['isin']}")
        if issues:
            inst["figi_verified"] = False
            inst["figi_source"] = "curated (UNVERIFIED: mismatch)"
            discrepancies.append((tk, "mismatch", "; ".join(issues)))
        else:
            inst["figi_verified"] = True
            inst["figi"] = ti["figi"]          # confirmed identical; mark authoritative source
            inst["figi_source"] = "t-invest"
    return discrepancies


def build_base_instruments(session: requests.Session) -> dict[str, dict]:
    """ISS metadata + curated (unverified) FIGI for every name."""
    instruments: dict[str, dict] = {}
    for tk in UNIVERSE:
        meta = fetch_iss_meta(session, tk)
        instruments[tk] = {
            "ticker": tk,
            "figi": CURATED_FIGI.get(tk),
            "figi_verified": False,
            "figi_source": "curated",
            **meta,
        }
    return instruments


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify-tinvest", action="store_true",
                    help="reconcile FIGI/lot/ISIN against live T-Invest (TINVEST_TOKEN from .env)")
    ap.add_argument("--tinvest-dump", default=None,
                    help="reconcile offline against a saved T-Invest instruments JSON export")
    args = ap.parse_args(argv)

    session = requests.Session()
    session.headers["User-Agent"] = "moex-instrument-metadata/0.2"
    instruments = build_base_instruments(session)

    tinvest: dict[str, dict] = {}
    source = "moex-iss (lot/step/isin) + curated figi (UNVERIFIED)"
    if args.verify_tinvest:
        tinvest = fetch_tinvest_shares(read_env_token())
        source = "moex-iss (lot/step/isin) + t-invest (figi, reconciled live)"
    elif args.tinvest_dump:
        tinvest = load_tinvest_dump(Path(args.tinvest_dump))
        source = "moex-iss (lot/step/isin) + t-invest (figi, reconciled from dump)"

    discrepancies: list[tuple[str, str, str]] = []
    if tinvest:
        discrepancies = reconcile_with_tinvest(instruments, tinvest)

    for tk, inst in instruments.items():
        flag = "VERIFIED" if inst["figi_verified"] else "unverified"
        print(f"  {tk:>6}  figi={inst['figi']}  lot={inst['lot']:>4}  "
              f"step={inst['min_price_step']}  isin={inst['isin']}  [{flag}]")

    all_verified = all(v["figi_verified"] for v in instruments.values())
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "all_figis_verified": all_verified,
        "instruments": instruments,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nWrote {len(instruments)} instruments -> {OUT_PATH}")
    if discrepancies:
        print(f"\n!! {len(discrepancies)} DISCREPANCY(IES) -- NOT auto-fixed, resolve manually:")
        for tk, kind, detail in discrepancies:
            print(f"   [{kind:8}] {tk}: {detail}")
    print(f"\nall_figis_verified = {all_verified}")
    if tinvest:
        n_ok = sum(1 for v in instruments.values() if v["figi_verified"])
        print(f"reconciled {n_ok}/{len(instruments)} names against T-Invest")
    # non-zero exit when a reconciliation was requested but did not fully pass
    return 0 if (all_verified or not tinvest) else 2


if __name__ == "__main__":
    raise SystemExit(main())
