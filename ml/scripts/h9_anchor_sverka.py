"""H9 step 4 — ANCHOR sverka: certify the forward feed's record/ex dates line up with the anchor
the research validated on, so the deployed sleeve trades the SAME event the edge was measured on.

Why this matters
    The research (`h9_dividend_research.py`) anchored every event on the RECORD date from MOEX ISS
    (`data/raw/dividends.csv` column `date` = registryclosedate), entered -12 TD and exited -2 TD —
    BEFORE the ex-gap. The live sleeve must anchor on the SAME object. The LLM forward feed
    (`data/news/dividend_calendar_upcoming.csv`) carries BOTH `record_date` and `ex_date`; if the
    feed's `record_date` were actually the ex-date (an easy mistake), the whole entry/exit window
    would slide by one trading day and we'd risk eating the ex-gap. This script proves it doesn't.

Checks (all must PASS):
  1. dividends.csv anchor column = `date` (record/registry-close); feed has `record_date` + `ex_date`.
  2. Feed internal consistency: ex_date = record_date - 1 trading day (T+1) for EVERY row.
  3. Merge correctness: load_dividend_calendar()'s FUTURE rows == feed `record_date` per ticker.
  4. Exit-before-ex invariant: deployed EXIT_OFFSET (-2 from record) sits strictly before ex (-1 from
     record under T+1), so the ex-gap is never captured.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ML_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = ML_DIR.parent
sys.path.insert(0, str(ML_DIR))

from src.service.dividend_sleeve import (  # noqa: E402
    load_dividend_calendar, ENTRY_OFFSET, EXIT_OFFSET, UPCOMING_FEED,
)

DATA_RAW = REPO_ROOT / "data" / "raw"
OUT = REPO_ROOT / "data" / "reports" / "h9_anchor_sverka.txt"


def main() -> int:
    lines: list[str] = []

    def pr(s: str = "") -> None:
        print(s)
        lines.append(s)

    hist = pd.read_csv(DATA_RAW / "dividends.csv")
    feed = pd.read_csv(UPCOMING_FEED)
    ok = True

    pr("H9 anchor sverka — does the forward feed trade the SAME anchor the research validated on?")
    pr("=" * 78)

    # --- Check 1: schema / semantic identity of the anchor column ----------------------------------
    c1 = "date" in hist.columns and {"record_date", "ex_date"}.issubset(feed.columns)
    ok &= c1
    pr(f"\n[1] anchor columns present  ->  {'PASS' if c1 else 'FAIL'}")
    pr(f"    dividends.csv: {list(hist.columns)}  (anchor = 'date' = ISS registryclosedate = RECORD)")
    pr(f"    feed: record_date, ex_date present = {set(['record_date','ex_date']).issubset(feed.columns)}")
    pr("    -> research anchored on RECORD date; sleeve merges feed.record_date -> 'date'. Same object.")

    # --- Check 2: feed internal T+1 consistency (ex = record - 1 trading day) -----------------------
    rec = pd.to_datetime(feed["record_date"]).to_numpy("datetime64[D]")
    exd = pd.to_datetime(feed["ex_date"]).to_numpy("datetime64[D]")
    gap = np.busday_count(exd, rec)            # trading days in [ex, record)
    bad = feed.loc[gap != 1, ["ticker", "record_date", "ex_date"]]
    c2 = len(bad) == 0
    ok &= c2
    pr(f"\n[2] ex_date == record_date - 1 trading day (T+1)  ->  {'PASS' if c2 else 'FAIL'}")
    pr(f"    {(gap == 1).sum()}/{len(feed)} rows consistent; trading-day gap values = {sorted(set(gap.tolist()))}")
    if not c2:
        for _, r in bad.iterrows():
            pr(f"    VIOLATION {r['ticker']}: record {r['record_date']} ex {r['ex_date']}")

    # --- Check 3: merge correctness — future calendar rows equal the feed's record_date -------------
    cal = load_dividend_calendar()
    hist_max = pd.to_datetime(hist["date"]).max()
    fut = cal[cal["date"] > pd.Timestamp(hist_max, tz="Europe/Moscow")]
    feed_map = {(r["ticker"], pd.Timestamp(r["record_date"]).date()) for _, r in feed.iterrows()}
    cal_map = {(t, d.date()) for t, d in zip(fut["ticker"], fut["date"])}
    missing = feed_map - cal_map
    c3 = len(missing) == 0 and len(fut) >= len(feed)
    ok &= c3
    pr(f"\n[3] merged calendar's FUTURE rows == feed record_date  ->  {'PASS' if c3 else 'FAIL'}")
    pr(f"    feed events {len(feed)}; merged future rows {len(fut)}; feed (ticker,record) all present "
       f"in merged = {len(missing) == 0}")
    if missing:
        pr(f"    MISSING from merged calendar: {sorted(missing)}")

    # --- Check 4: exit-before-ex invariant ---------------------------------------------------------
    # research/sleeve anchor on record; ex is at offset -1 (T+1); we exit at -EXIT_OFFSET.
    c4 = EXIT_OFFSET >= 2  # exit at record-2 = ex-1 (or earlier) -> strictly before the ex-gap
    ok &= c4
    pr(f"\n[4] exit sits before the ex-gap  ->  {'PASS' if c4 else 'FAIL'}")
    pr(f"    anchor=record; ex=record-1 TD (T+1); entry=record-{ENTRY_OFFSET}, exit=record-{EXIT_OFFSET}")
    pr(f"    last held return is INTO record-{EXIT_OFFSET} (= ex-{EXIT_OFFSET-1}); ex-gap at record-1 "
       f"NOT captured -> {'OK' if c4 else 'EX-GAP RISK'}")
    pr("    note: the sleeve uses ONLY record_date as the anchor; ex_date is informational (not traded).")

    pr("\n" + "=" * 78)
    pr(f"VERDICT: {'PASS — feed anchor == research anchor; deployed window trades the right event.' if ok else 'FAIL — anchor mismatch, fix before trading.'}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n-> {OUT}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
