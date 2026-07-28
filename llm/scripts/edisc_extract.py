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

Anti-bot posture (2026-07-28): the WAF started answering with an INTERACTIVE challenge ("разверните
картинку горизонтально") whose own text blames browsing/clicking speed. Two honest responses, both
implemented here — no fingerprint spoofing, no challenge solving:
  1. PACE. Randomised delays between result pages and between tickers (was 0.3s and none). This runs
     by hand a couple of times a month; there is no reason to hammer the site.
  2. HUMAN IN THE LOOP. `--headed` opens a real window so a person clears the challenge themselves,
     and a PERSISTENT PROFILE keeps that session so later unattended runs reuse it. If a headless run
     is challenged it stops and says to re-run with --headed, rather than pretending it pulled data.

Run (smoke, one ticker):
  & "ml\.venv-win\Scripts\python.exe" llm\scripts\edisc_extract.py --tickers SBER
Run (full universe):
  & "ml\.venv-win\Scripts\python.exe" llm\scripts\edisc_extract.py
First run / after a block — clear the challenge by hand (a window opens):
  & "ml\.venv-win\Scripts\python.exe" llm\scripts\edisc_extract.py --headed --tickers SBER
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import sys
import time
from pathlib import Path
from urllib.parse import urlencode

from playwright.sync_api import sync_playwright

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")
SEARCH_URL = "https://www.e-disclosure.ru/poisk-po-soobshheniyam"
OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "news" / "edisclosure"

# Persistent browser profile: cookies survive between runs, so a challenge cleared ONCE by a human
# (--headed) lets later unattended runs reuse that session instead of tripping the check every time.
PROFILE_DIR = Path(__file__).resolve().parents[2] / "data" / "news" / ".edisc_profile"

FORM_SEL = "#sEventSearchForm"       # only present on the real search page
CHALLENGE_MARK = "xpvnsulc"          # ServicePipe interstitial path
# How long to let a human solve the challenge in --headed mode.
CHALLENGE_WAIT_MS = 5 * 60 * 1000

# issuer_key -> (companyID to keep, name substring to query). Pin by companyID, not text. One parquet
# per issuer; prefs (SBERP/SNGSP/TATNP) share their ordinary issuer's pull (same companyID).
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
    # H9 universe expansion 2026-06-21 — new issuers (companyIDs discovered 2026-06-21).
    "SIBN": (347,  "Газпром нефть"),      # ПАО «Газпром нефть» (distinct from parent GAZP id 934)
    "PHOR": (573,  "ФосАгро"),            # ПАО «ФосАгро»
    "MOEX": (43,   "Московская Биржа"),   # ПАО Московская Биржа
}

START_DATE = dt.date(2020, 1, 1)          # candle history starts 2020-01-03
PAGE_SIZE = 100
MAX_PAGES = 20            # 20*100 = 2000 > server cap (~1200); used to drain a window
RESULT_CAP = 1200         # server truncates a query at ~1200 newest rows -> must subdivide

# --- pacing -------------------------------------------------------------------------------------
# The WAF blocked this scraper on 2026-07-28 with an interactive challenge whose own text names the
# trigger: "просматриваете страницы и кликаете со скоростью". The old pacing was 0.30s between API
# pages and NO gap between tickers — 19 page loads back to back. These delays are the fix: be a slow,
# obviously-non-abusive client. Randomised so the pattern isn't a metronome. Slower is fine — this
# runs at most a couple of times a month, by hand.
PAUSE_S = (1.0, 2.2)          # between API result pages
TICKER_PAUSE_S = (6.0, 14.0)  # between tickers (each starts with a fresh page load)


def _nap(span: tuple[float, float]) -> None:
    time.sleep(random.uniform(*span))


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
        _nap(PAUSE_S)
        raw.extend(items)
        if len(items) < PAGE_SIZE:
            break
    return raw, len(raw) >= RESULT_CAP


def open_search(pg, headed: bool, first: bool = False) -> bool:
    """Load the search page, clearing the anti-bot challenge if one is shown. True = form is ready.

    The WAF may answer with the ServicePipe interstitial instead of the page. It is an INTERACTIVE
    check ("разверните картинку горизонтально") that never resolves on its own, so:
      * headed  -> a human solves it once; we simply wait for the form to appear, and the persistent
                   profile keeps that cookie for subsequent (even headless) runs.
      * headless-> we do NOT try to defeat the check. We report it and tell the operator to re-run
                   with --headed, which is the supported way through.
    """
    pg.goto(SEARCH_URL, wait_until="domcontentloaded", timeout=60000)
    try:
        pg.wait_for_selector(FORM_SEL, timeout=15000 if first else 25000)
        return True
    except Exception:
        pass

    challenged = CHALLENGE_MARK in pg.url or not pg.locator(FORM_SEL).count()
    if not challenged:
        return False
    if not headed:
        print("\n!! anti-bot challenge — the search form never appeared.\n"
              f"   landed on: {pg.url[:120]}\n"
              "   This is an INTERACTIVE check; it cannot clear itself and is not bypassed here.\n"
              "   Re-run once with --headed and solve it by hand; the profile then keeps the\n"
              f"   session ({PROFILE_DIR}) so later runs go through unattended.", flush=True)
        return False

    print("\n>> anti-bot challenge shown. Solve it in the open browser window "
          f"(waiting up to {CHALLENGE_WAIT_MS // 60000} min)...", flush=True)
    try:
        pg.wait_for_selector(FORM_SEL, timeout=CHALLENGE_WAIT_MS)
        print(">> challenge cleared — session stored in the profile.", flush=True)
        return True
    except Exception:
        print("!! challenge not cleared in time.", flush=True)
        return False


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
    ap.add_argument("--headed", action="store_true",
                    help="open a real browser window so a human can clear the anti-bot challenge; "
                         "the session is kept in the persistent profile for later unattended runs")
    ap.add_argument("--profile", default=str(PROFILE_DIR),
                    help=f"persistent browser profile dir (default {PROFILE_DIR})")
    args = ap.parse_args()
    start_date = dt.date.fromisoformat(args.since) if args.since else START_DATE

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        import pandas as pd
    except Exception as exc:
        print("pandas required:", exc)
        return 1

    profile = Path(args.profile)
    profile.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        # Persistent context (not launch()+new_context()): the WAF cookie earned by a --headed run
        # must outlive the process, otherwise every run starts as an unknown client and gets checked.
        ctx = p.chromium.launch_persistent_context(
            str(profile), headless=not args.headed, locale="ru-RU", user_agent=UA,
            viewport={"width": 1400, "height": 900})
        pg = ctx.pages[0] if ctx.pages else ctx.new_page()
        end_date = dt.date.today()  # never request a future dateFinish (returns empty)

        blocked = False
        for i, tk in enumerate(args.tickers):
            if tk not in UNIVERSE:
                print(f"skip unknown ticker {tk}")
                continue
            if i:
                _nap(TICKER_PAUSE_S)   # pace the run; each ticker starts with a fresh page load
            # refresh the WAF cookie per ticker so a long run can't silently expire mid-stream
            if not open_search(pg, args.headed, first=(i == 0)):
                blocked = True
                break
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
        ctx.close()

    if blocked:
        # Non-zero so the refresh orchestrator marks the run `degraded` and keeps the last-good feed,
        # instead of silently reporting a successful pull that discovered nothing.
        print("\nextract STOPPED at the anti-bot challenge — tickers already pulled were saved; "
              "re-run with --headed to clear it.", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
