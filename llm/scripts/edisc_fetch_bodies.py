# -*- coding: utf-8 -*-
r"""Fetch + cache e-disclosure.ru substantial-fact message BODIES for the dividend chain.

The parquet snapshots in data/news/edisclosure/ store only the TITLE of each disclosure; the
dividend amount and the record date live in the message BODY. The body page is reached at
    https://www.e-disclosure.ru/portal/event.aspx?EventId=<pseudoGUID>
(cracked 2026-06-19 — this was the open item #4 in llm/docs/NEWS_SOURCE_EDISCLOSURE.md). The page
is behind the same ServicePipe WAF as the search API, so we drive it with Playwright/Chromium and
read inner_text("body"); we keep only the message core (between the "код сообщения" marker and the
"Версия для печати" footer).

We fetch the bodies of the dividend-chain title classes (board decision / AGM convocation / AGM
decision / explicit dividend-recommendation press release) published since FETCH_FLOOR, and cache
each body to data/news/edisclosure_bodies/<TICKER>_<guid>.txt so the parser can run offline and
reruns are cheap. No-lookahead is unaffected: bodies are keyed by the same pubDate as the title.

Run: $env:PYTHONIOENCODING="utf-8"; & "ml\.venv-win\Scripts\python.exe" llm\scripts\edisc_fetch_bodies.py
     (optional: --tickers LKOH CHMF   --since 2026-02-01   --refetch)
"""
from __future__ import annotations

import argparse
import io
import re
import sys
import time
from pathlib import Path

import pandas as pd
from playwright.sync_api import sync_playwright

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

REPO = Path(__file__).resolve().parents[2]
DDIR = REPO / "data" / "news" / "edisclosure"
BODY_DIR = REPO / "data" / "news" / "edisclosure_bodies"

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")
SEARCH_URL = "https://www.e-disclosure.ru/poisk-po-soobshheniyam"
EVENT_URL = "https://www.e-disclosure.ru/portal/event.aspx?EventId="

# e-disclosure issuer files to scan for dividend-chain bodies. Prefs (SBERP/SNGSP/TATNP) are NOT
# listed — they share their ordinary issuer's disclosures; SIBN/PHOR/MOEX are new issuers (2026-06-21).
UNIVERSE = ("SBER GAZP LKOH GMKN ROSN NVTK TATN MGNT MTSS SNGS CHMF ALRS VTBR MAGN NLMK PLZL "
            "SIBN PHOR MOEX").split()

# dividend-chain title classes whose bodies can carry the dividend amount / record date
RE_BOARD = re.compile(r"решени[яй] совета директоров|решени.*наблюдательн", re.I)
RE_CONV = re.compile(r"созыв общего собрания|о проведении.*общего собрания акционеров", re.I)
RE_AGM = re.compile(r"решени[яй] общ(?:их|его) собран", re.I)
RE_DIV_PR = re.compile(r"рекомендац\w*.{0,40}дивиденд|дивиденд\w*.{0,40}рекомендац", re.I)
DIV_CHAIN = re.compile("|".join(x.pattern for x in (RE_BOARD, RE_CONV, RE_AGM, RE_DIV_PR)), re.I)

FETCH_FLOOR = "2026-02-01"  # FY2025 dividend cycle: board recos land Feb-May 2026
PAUSE_S = 0.4


def candidates(tickers: list[str], since: str) -> pd.DataFrame:
    floor = pd.Timestamp(since)
    rows = []
    for t in tickers:
        p = DDIR / f"{t}.parquet"
        if not p.exists():
            print(f"  [warn] no parquet for {t}")
            continue
        d = pd.read_parquet(p, columns=["event_name", "pub_date", "event_date", "pseudo_guid"])
        d["pub"] = pd.to_datetime(d["pub_date"], format="ISO8601").dt.tz_localize(None)
        d = d[(d["pub"] >= floor) & d["event_name"].astype(str).str.contains(DIV_CHAIN)]
        for _, r in d.iterrows():
            rows.append((t, r["pub"], str(r["event_name"]), str(r["event_date"])[:10], r["pseudo_guid"]))
    return pd.DataFrame(rows, columns=["ticker", "pub", "event_name", "event_date", "guid"])


def body_path(ticker: str, guid: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", guid)
    return BODY_DIR / f"{ticker}_{safe}.txt"


def extract_core(txt: str) -> str:
    s = txt.find("код сообщ")
    e = txt.find("Версия для печати", s + 10 if s >= 0 else 0)
    return txt[s:e].strip() if s >= 0 and e > s else txt.strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="*", default=UNIVERSE)
    ap.add_argument("--since", default=FETCH_FLOOR)
    ap.add_argument("--refetch", action="store_true", help="re-fetch even if cached")
    args = ap.parse_args()

    BODY_DIR.mkdir(parents=True, exist_ok=True)
    cand = candidates(args.tickers, args.since)
    todo = cand if args.refetch else cand[~cand.apply(
        lambda r: body_path(r["ticker"], r["guid"]).exists(), axis=1)]
    print(f"candidates: {len(cand)} | to fetch: {len(todo)} | cached: {len(cand) - len(todo)}")
    if todo.empty:
        return 0

    with sync_playwright() as p:
        b = p.chromium.launch(headless=True)
        pg = b.new_context(locale="ru-RU", user_agent=UA).new_page()
        pg.goto(SEARCH_URL, wait_until="domcontentloaded", timeout=60000)
        pg.wait_for_selector("#sEventSearchForm", timeout=45000)
        pg.wait_for_timeout(1000)

        ok = fail = 0
        for i, (_, r) in enumerate(todo.iterrows(), 1):
            try:
                pg.goto(EVENT_URL + r["guid"], wait_until="domcontentloaded", timeout=40000)
                core = extract_core(pg.inner_text("body"))
                if len(core) < 100:
                    raise RuntimeError(f"body too short ({len(core)})")
                body_path(r["ticker"], r["guid"]).write_text(core, encoding="utf-8")
                ok += 1
                if i % 10 == 0 or i == len(todo):
                    print(f"  {i}/{len(todo)} fetched (ok={ok} fail={fail})", flush=True)
            except Exception as exc:
                fail += 1
                print(f"  [fail] {r['ticker']} {r['pub'].date()} {r['guid'][:12]}: {exc}")
            time.sleep(PAUSE_S)
            if i % 25 == 0:  # refresh WAF cookie on long runs
                pg.goto(SEARCH_URL, wait_until="domcontentloaded", timeout=60000)
                pg.wait_for_timeout(800)
        b.close()
        print(f"done: ok={ok} fail={fail} | cache dir {BODY_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
