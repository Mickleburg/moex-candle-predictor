# -*- coding: utf-8 -*-
r"""Extract e-disclosure.ru disclosure history for the 12-ticker universe.

Mechanism (validated 2026-06-15, see llm/docs/NEWS_SOURCE_EDISCLOSURE.md):
  - Playwright/Chromium passes the ServicePipe anti-bot WAF.
  - Call POST /api/search/sevents via IN-PAGE fetch (carries the WAF cookie).
  - Server filters only by company-name substring (`query`); there is NO companyId
    filter, so we filter the returned rows client-side by the target companyID.
  - Iterate yearly windows to keep page counts small even for broad names
    (GAZP/ROSN/MGNT have many same-name subsidiaries).

No-lookahead: we keep `pubDate` (publication time). `eventDate` is stored for
reference only and MUST NOT be used as the as-of time downstream.

Run (smoke, one ticker):
  & "ml\.venv-win\Scripts\python.exe" llm\scripts\edisc_extract.py --tickers SBER
Run (full universe):
  & "ml\.venv-win\Scripts\python.exe" llm\scripts\edisc_extract.py
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path
from urllib.parse import urlencode

from playwright.sync_api import sync_playwright

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")
SEARCH_URL = "https://www.e-disclosure.ru/poisk-po-soobshheniyam"
OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "news" / "edisclosure"

# ticker -> (companyID to keep, name substring to query). Pin by companyID, not text.
UNIVERSE: dict[str, tuple[int, str]] = {
    "SBER": (3043, "Сбербанк"),
    "GAZP": (934,  "Газпром"),            # 347=Газпром нефть is ticker SIBN, excluded by id
    "LKOH": (17,   "ЛУКОЙЛ"),
    "GMKN": (564,  "Норильский никель"),
    "ROSN": (6505, "Роснефть"),
    "NVTK": (225,  "НОВАТЭК"),
    "TATN": (118,  "Татнефть"),
    "MGNT": (7671, "Магнит"),             # duplicate ПАО Магнит 9581 — verify listed issuer
    "MTSS": (236,  "Мобильные ТелеСистемы"),
    "SNGS": (312,  "Сургутнефтегаз"),
    "CHMF": (30,   "Северсталь"),
    "ALRS": (199,  "АЛРОСА"),
    # extended dividend-certification universe (companyIDs discovered 2026-06-17)
    "VTBR": (1210, "ВТБ"),                # Банк ВТБ (ПАО); query broad -> client-side id filter
    "MAGN": (9,    "Магнитогорский металлургический"),   # ПАО "ММК"
    "NLMK": (2509, "Новолипецкий металлургический"),     # ПАО "НЛМК"
    "PLZL": (7832, "Полюс"),              # ПАО «Полюс»
}

START_DATE = dt.date(2020, 1, 1)          # candle history starts 2020-01-03
PAGE_SIZE = 100
MAX_PAGES = 20            # 20*100 = 2000 > server cap (~1200); used to drain a window
RESULT_CAP = 1200         # server truncates a query at ~1200 newest rows -> must subdivide
PAUSE_S = 0.30            # polite delay between requests


def _fmt(d: dt.date) -> str:
    return d.strftime("%d.%m.%Y")


def _body(query: str, d0: dt.date, d1: dt.date, page: int) -> str:
    return urlencode({
        "eventTypeTerm": "", "radView": "0",
        "dateStart": _fmt(d0), "dateFinish": _fmt(d1),
        "textfieldEvent": "", "radReg": "FederalDistricts",
        "districtsCheckboxGroup": "-1", "regionsCheckboxGroup": "-1",
        "branchesCheckboxGroup": "-1",
        "textfieldCompany": query, "query": query,
        "lastPageSize": str(PAGE_SIZE), "lastPageNumber": str(page),
        "queryEvent": "",
    })


def _fetch_page(pg, query: str, d0: dt.date, d1: dt.date, page: int) -> list[dict]:
    txt = pg.evaluate(
        """async (bd) => {
            const r = await fetch('/api/search/sevents', {method:'POST',
              headers:{'Content-Type':'application/x-www-form-urlencoded; charset=UTF-8',
                       'X-Requested-With':'XMLHttpRequest'}, body: bd});
            return await r.text();
        }""", _body(query, d0, d1, page))
    try:
        return json.loads(txt).get("foundEventsList", []) or []
    except Exception:
        return []


def _drain_window(pg, query: str, d0: dt.date, d1: dt.date) -> tuple[list[dict], bool]:
    """Return (all raw rows for [d0,d1], truncated?). Truncated => hit the result cap."""
    raw: list[dict] = []
    for page in range(1, MAX_PAGES + 1):
        items = _fetch_page(pg, query, d0, d1, page)
        time.sleep(PAUSE_S)
        raw.extend(items)
        if len(items) < PAGE_SIZE:
            break
    return raw, len(raw) >= RESULT_CAP


def extract_ticker(pg, ticker: str, company_id: int, query: str,
                   start_date: dt.date, end_date: dt.date) -> list[dict]:
    """Adaptive date-range split: subdivide any window that hits the server cap so no
    rows are silently dropped. dateFinish is clamped to today by the caller (end_date)."""
    seen: set[str] = set()
    rows: list[dict] = []

    def recurse(d0: dt.date, d1: dt.date) -> None:
        raw, truncated = _drain_window(pg, query, d0, d1)
        if truncated and d0 < d1:
            mid = d0 + (d1 - d0) // 2
            recurse(d0, mid)
            recurse(mid + dt.timedelta(days=1), d1)
            return
        for e in raw:
            if e.get("companyID") != company_id:
                continue
            guid = e.get("pseudoGUID")
            if guid in seen:
                continue
            seen.add(guid)
            rows.append({
                "ticker": ticker,
                "company_id": company_id,
                "company_name": e.get("companyName"),
                "event_name": e.get("eventName"),
                "pub_date": e.get("pubDate"),       # publication time (no-lookahead key)
                "event_date": e.get("eventDate"),   # reference only — DO NOT use as as-of
                "pseudo_guid": guid,
                "agency": e.get("agency"),
                "corrected": e.get("isCorrectedByAnotherEvent"),
            })

    recurse(start_date, end_date)
    print(f"  [{ticker}] done: kept {len(rows)} rows", flush=True)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="*", default=list(UNIVERSE),
                    help="subset of tickers (default: all 12)")
    ap.add_argument("--since", default=None,
                    help="pull window start YYYY-MM-DD (default 2020-01-01; use a recent date for "
                         "a light incremental EOD update)")
    ap.add_argument("--merge", action="store_true",
                    help="merge the pulled window into the existing parquet (dedup by pseudo_guid, "
                         "keep history) instead of overwriting — for incremental refresh")
    args = ap.parse_args()
    start_date = dt.date.fromisoformat(args.since) if args.since else START_DATE

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        import pandas as pd
    except Exception as exc:
        print("pandas required:", exc)
        return 1

    with sync_playwright() as p:
        b = p.chromium.launch(headless=True)
        ctx = b.new_context(locale="ru-RU", user_agent=UA)
        pg = ctx.new_page()
        end_date = dt.date.today()  # never request a future dateFinish (returns empty)

        for tk in args.tickers:
            if tk not in UNIVERSE:
                print(f"skip unknown ticker {tk}")
                continue
            # refresh the WAF cookie per ticker so a long run can't silently expire mid-stream
            pg.goto(SEARCH_URL, wait_until="domcontentloaded", timeout=60000)
            pg.wait_for_selector("#sEventSearchForm", timeout=45000)
            pg.wait_for_timeout(1200)

            cid, query = UNIVERSE[tk]
            print(f"=== {tk} (companyID={cid}, query='{query}', since={start_date}) ===", flush=True)
            rows = extract_ticker(pg, tk, cid, query, start_date, end_date)
            df = pd.DataFrame(rows)
            out = OUT_DIR / f"{tk}.parquet"
            if args.merge and out.exists():
                prior = pd.read_parquet(out)
                before = len(prior)
                df = pd.concat([prior, df], ignore_index=True)
                df = df.drop_duplicates(subset=["pseudo_guid"], keep="last")
                added = len(df) - before
            else:
                added = len(df)
            if not df.empty:
                df = df.sort_values("pub_date").reset_index(drop=True)
            df.to_parquet(out, index=False)
            print(f"--> wrote {out}  rows={len(df)} (+{added} new)", flush=True)
        b.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
