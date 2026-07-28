"""ISS dividend-history backfill for the H9 sleeve -- past-data completion, no lookahead.

Closes two gaps (ml/docs/research/h9_universe_expansion_2026-06-21.md):
* the **10-month hole** in ``data/raw/dividends.csv`` (history ended 2025-07-20); and
* **full history for the new expansion lines** (SBERP SNGSP PHOR MOEX -- pref payouts differ
  from the ordinary, so each line's own series is fetched).

Source: MOEX ISS ``/iss/securities/{SECID}/dividends.json`` -> columns
``secid, isin, registryclosedate, value, currencyid``. ``registryclosedate`` is the RECORD
date, which is exactly the anchor the sleeve uses (``dividends.csv`` column ``date``).

Discipline (provenance + no silent edits):
* These are events that have **already occurred** -- pure past data, no lookahead.
* Existing rows are NEVER overwritten: on a ``(ticker, date)`` clash the stored value wins
  (preserves the research-validated history); ISS value differences are reported, not applied.
* Every row carries a ``source`` column; newly added rows get ``iss_backfill_<rundate>``.
* A provenance sidecar (``dividends_provenance.json``) records ranges per source and FLAGS the
  **2025-08..present CORROBORATION window** (it overlaps the burned 2025-09..2026-06 split) so
  ML/lead segment it before any forward use -- this script does not segment, it just delivers
  clean dated data.

Usage::

    python -m backend.dividends                      # backfill ALL universe lines
    python -m backend.dividends --tickers SBER PHOR   # a subset
"""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import requests

from .universe import EXPANSION_SHARES, SHARES

_REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = _REPO_ROOT / "data" / "raw"
CSV_PATH = DATA_RAW / "dividends.csv"
PROVENANCE_PATH = DATA_RAW / "dividends_provenance.json"
MOEX_ISS_BASE = "https://iss.moex.com"

# Window that overlaps the burned 2025-09..2026-06 test split -> corroboration, not forward.
CORROBORATION_START = "2025-08-01"

ALL_UNIVERSE = tuple(SHARES) + tuple(EXPANSION_SHARES)


def fetch_iss_dividends(secid: str, session: requests.Session) -> pd.DataFrame:
    """Full ISS dividend history for one SECID -> [ticker, date(record), value, ccy]."""
    url = f"{MOEX_ISS_BASE}/iss/securities/{secid}/dividends.json"
    resp = session.get(url, params={"iss.meta": "off"}, timeout=20)
    resp.raise_for_status()
    block = resp.json()["dividends"]
    cols = {c: i for i, c in enumerate(block["columns"])}
    rows = []
    for r in block["data"]:
        rdate = r[cols["registryclosedate"]]
        value = r[cols["value"]]
        if not rdate or value in (None, "", 0):
            continue
        rows.append({
            "ticker": secid,
            "date": str(rdate),
            "value": float(value),
            "ccy": r[cols["currencyid"]] or "RUB",
        })
    return pd.DataFrame(rows, columns=["ticker", "date", "value", "ccy"])


def _load_existing(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        return pd.DataFrame(columns=["ticker", "date", "value", "ccy", "source"])
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    if "source" not in df.columns:
        df["source"] = "iss_history"   # pre-existing validated ISS history
    return df


def backfill(tickers=ALL_UNIVERSE, csv_path: Path = CSV_PATH,
             session: Optional[requests.Session] = None,
             run_date: Optional[str] = None,
             fetch_fn: Optional[Callable[[str], pd.DataFrame]] = None,
             provenance_path: Path = PROVENANCE_PATH) -> dict:
    """Fetch ISS dividends for ``tickers`` and merge into ``dividends.csv`` (existing wins).

    Returns a machine-readable report (added rows, discrepancies, provenance summary).
    ``fetch_fn(ticker) -> DataFrame[ticker,date,value,ccy]`` is injectable for tests; the
    default hits MOEX ISS.
    """
    run_date = run_date or datetime.now(timezone.utc).date().isoformat()
    src_tag = f"iss_backfill_{run_date}"
    if fetch_fn is None:
        session = session or requests.Session()
        session.headers.setdefault("User-Agent", "moex-dividend-backfill/0.1")
        fetch_fn = lambda tk: fetch_iss_dividends(tk, session)  # noqa: E731

    existing = _load_existing(csv_path)
    have = set(zip(existing["ticker"], existing["date"])) if not existing.empty else set()

    fetched = []
    for tk in tickers:
        fetched.append(fetch_fn(tk))
    iss = (pd.concat(fetched, ignore_index=True) if fetched
           else pd.DataFrame(columns=["ticker", "date", "value", "ccy"]))
    iss = iss.drop_duplicates(subset=["ticker", "date"], keep="last")

    # discrepancies: same (ticker,date), different value -> report, do NOT overwrite
    discrepancies = []
    if not existing.empty:
        ex_val = {(t, d): v for t, d, v in
                  zip(existing["ticker"], existing["date"], existing["value"])}
        for t, d, v in zip(iss["ticker"], iss["date"], iss["value"]):
            ov = ex_val.get((t, d))
            if ov is not None and abs(float(ov) - float(v)) > 1e-6:
                discrepancies.append({"ticker": t, "date": d, "stored": ov, "iss": v})

    new_rows = iss[[(t, d) not in have for t, d in zip(iss["ticker"], iss["date"])]].copy()
    new_rows["source"] = src_tag

    merged = pd.concat([existing, new_rows], ignore_index=True)
    merged = merged.drop_duplicates(subset=["ticker", "date"], keep="first")  # existing wins
    merged = merged.sort_values(["ticker", "date"]).reset_index(drop=True)
    merged.to_csv(csv_path, index=False)   # CSV: the ML reader loads this by column name

    report = _write_provenance(merged, new_rows, discrepancies, tickers, run_date,
                               src_tag, provenance_path)
    return report


def _write_provenance(merged, new_rows, discrepancies, tickers, run_date, src_tag,
                      provenance_path: Path = PROVENANCE_PATH) -> dict:
    def _range(df):
        return ["", ""] if df.empty else [df["date"].min(), df["date"].max()]

    corrob = new_rows[new_rows["date"] >= CORROBORATION_START] if not new_rows.empty else new_rows
    by_source = {s: int(n) for s, n in merged["source"].value_counts().items()}

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_date": run_date,
        "source_endpoint": "moex-iss /iss/securities/{SECID}/dividends.json (registryclosedate=record)",
        "tickers_fetched": list(tickers),
        "total_events": int(len(merged)),
        "rows_added_this_run": int(len(new_rows)),
        "rows_by_source": by_source,
        "full_range": _range(merged),
        "added_range": _range(new_rows),
        "value_discrepancies_reported_not_applied": discrepancies,
        "corroboration_window": {
            "note": ("events dated >= 2025-08-01 OVERLAP the burned 2025-09..2026-06 split; "
                     "they are CORROBORATION, not forward signal -- ML/lead must segment before "
                     "any forward use; do not mix into the forward feed unmarked."),
            "since": CORROBORATION_START,
            "rows_added_in_window": int(len(corrob)),
            "tickers_in_window": sorted(corrob["ticker"].unique().tolist()) if not corrob.empty else [],
        },
    }
    provenance_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def promote_events(events: pd.DataFrame, source: str = "e-disclosure",
                   csv_path: Path = CSV_PATH, run_date: Optional[str] = None,
                   provenance_path: Path = PROVENANCE_PATH) -> dict:
    """Upsert already-REALIZED dividend events from a NON-ISS source (the e-disclosure forward feed as
    its events realize) into ``dividends.csv`` -- the same merge discipline as :func:`backfill`.

    Why this exists: the forward feed is the ONLY source of a dividend's record date until MOEX ISS
    publishes it ~11 months later, and the feed builder keeps only a rolling forward window. Once an
    event's record date ages out of that window it is dropped from the feed and -- since ISS still
    lacks it -- disappears from ``load_dividend_calendar`` entirely, silently regressing the H9 gate's
    forward ``n``. Promoting realized events into the permanent history is what stops that loss.

    Discipline (identical to backfill): existing rows WIN (validated history / an earlier promotion is
    never overwritten); every new row carries ``source``; a value clash on an existing (ticker, date)
    is REPORTED, not applied. Idempotent -- re-promoting the same event adds nothing.

    ``events`` needs columns [ticker, date, value] (date = record date); ``ccy`` optional (RUB).
    """
    run_date = run_date or datetime.now(timezone.utc).date().isoformat()
    if events is None or len(events) == 0:
        return {"rows_added_this_run": 0, "source": source, "note": "no events to promote"}

    ev = events.copy()
    ev["date"] = pd.to_datetime(ev["date"]).dt.strftime("%Y-%m-%d")
    ev["value"] = pd.to_numeric(ev["value"], errors="coerce")
    if "ccy" not in ev.columns:
        ev["ccy"] = "RUB"
    ev = ev[["ticker", "date", "value", "ccy"]].dropna(subset=["ticker", "date", "value"])
    ev = ev[ev["value"] > 0].drop_duplicates(subset=["ticker", "date"], keep="last")

    existing = _load_existing(csv_path)
    have = set(zip(existing["ticker"], existing["date"])) if not existing.empty else set()

    # value clash on an already-known (ticker, date): report, never overwrite (history stays authoritative)
    discrepancies = []
    if not existing.empty:
        ex_val = {(t, d): v for t, d, v in
                  zip(existing["ticker"], existing["date"], existing["value"])}
        for t, d, v in zip(ev["ticker"], ev["date"], ev["value"]):
            ov = ex_val.get((t, d))
            if ov is not None and abs(float(ov) - float(v)) > 1e-6:
                discrepancies.append({"ticker": t, "date": d, "stored": ov, "incoming": v})

    new_rows = ev[[(t, d) not in have for t, d in zip(ev["ticker"], ev["date"])]].copy()
    new_rows["source"] = source

    if new_rows.empty and not discrepancies:
        # True no-op: touch NOTHING. Re-running a refresh must not rewrite dividends.csv nor bump the
        # provenance sidecar's `generated_at` — that churn made an otherwise byte-identical refresh
        # show up as a repo change and quietly broke the documented idempotency contract.
        return {"rows_added_this_run": 0, "source": source, "total_events": int(len(existing)),
                "value_discrepancies_reported_not_applied": [], "note": "no new events (no-op)"}

    merged = pd.concat([existing, new_rows], ignore_index=True)
    merged = merged.drop_duplicates(subset=["ticker", "date"], keep="first")   # existing wins
    merged = merged.sort_values(["ticker", "date"]).reset_index(drop=True)
    merged.to_csv(csv_path, index=False)

    report = _write_provenance(merged, new_rows, discrepancies,
                               sorted(ev["ticker"].unique().tolist()), run_date, source,
                               provenance_path)
    report["source"] = source
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tickers", nargs="+", default=list(ALL_UNIVERSE),
                    help="lines to backfill (default: full equity universe)")
    ap.add_argument("--csv", default=str(CSV_PATH))
    args = ap.parse_args(argv)

    report = backfill(tuple(args.tickers), Path(args.csv))
    print(f"Dividend backfill: +{report['rows_added_this_run']} rows "
          f"-> {report['total_events']} total events")
    print(f"  added range: {report['added_range'][0]} .. {report['added_range'][1]}")
    print(f"  by source:   {report['rows_by_source']}")
    cw = report["corroboration_window"]
    print(f"  corroboration (>= {cw['since']}): {cw['rows_added_in_window']} rows, "
          f"tickers={cw['tickers_in_window']}")
    if report["value_discrepancies_reported_not_applied"]:
        print(f"  !! {len(report['value_discrepancies_reported_not_applied'])} value discrepancy(ies) "
              f"(stored kept, NOT overwritten):")
        for d in report["value_discrepancies_reported_not_applied"][:10]:
            print(f"     {d['ticker']} {d['date']}: stored={d['stored']} iss={d['iss']}")
    print(f"  provenance -> {PROVENANCE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
