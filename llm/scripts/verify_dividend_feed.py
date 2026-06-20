# -*- coding: utf-8 -*-
r"""Independent no-lookahead verifier for the forward dividend feed.

This re-checks `data/news/dividend_calendar_upcoming.csv` from the RAW disclosure pub_dates (the
parquet snapshots) instead of trusting the columns the builder wrote — a second, independent pair of
eyes the scheduled refresh runs as a gate before the new CSV is allowed to replace the live one. It
deliberately re-implements its own title-class regexes so a bug in the builder's parser cannot also
pass the verifier.

Invariants (all must hold for every feed row):
  1. board_reco_date is a REAL disclosure pubDate — some board / AGM-convocation / dividend-reco
     disclosure for that ticker was published on exactly that calendar day (date not fabricated).
  2. board_reco_date <= as_of                — no future disclosure was used (no-lookahead).
  3. board_reco_date <= record_date - 12 TD  — the dividend was known early enough to enter 12 TD
     ahead (this is the whole point of the feed; the entry deadline is not in the past at reco time).
  4. source_url's disclosure pubDate <= as_of — the cited authoritative event isn't future-dated.
  5. ex_date == record_date - 1 trading day  — matches the ML anchor sverka (weekend-only busday).
  6. value parses as a float > 0             — else ML's load_dividend_calendar drops the row silently.

Run: $env:PYTHONIOENCODING="utf-8"; & "ml\.venv-win\Scripts\python.exe" llm\scripts\verify_dividend_feed.py
     (exit 0 = all pass; exit 1 = at least one violation. --as-of YYYY-MM-DD overrides today.)
"""
from __future__ import annotations

import argparse
import io
import re
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

REPO = Path(__file__).resolve().parents[2]
DDIR = REPO / "data" / "news" / "edisclosure"
FEED_CSV = REPO / "data" / "news" / "dividend_calendar_upcoming.csv"
ENTRY_LEAD_TD = 12

# independent (re-implemented) title classes that can carry a board dividend recommendation
RE_RECO_TITLE = re.compile(
    r"решени[яй] совета директоров|решени.*наблюдательн"
    r"|созыв общего собрания|о проведении.*общего собрания акционеров"
    r"|рекомендац\w*.{0,40}дивиденд|дивиденд\w*.{0,40}рекомендац", re.I)


def _load_pub(ticker: str) -> pd.DataFrame:
    p = DDIR / f"{ticker}.parquet"
    if not p.exists():
        return pd.DataFrame(columns=["event_name", "pub", "pseudo_guid"])
    d = pd.read_parquet(p, columns=["event_name", "pub_date", "pseudo_guid"])
    d["pub"] = pd.to_datetime(d["pub_date"], format="ISO8601").dt.tz_localize(None)
    return d


def verify(feed: pd.DataFrame, as_of: pd.Timestamp) -> tuple[bool, list[str], dict]:
    lines: list[str] = []
    viol = 0
    checked = 0
    cache: dict[str, pd.DataFrame] = {}

    def pub(t: str) -> pd.DataFrame:
        if t not in cache:
            cache[t] = _load_pub(t)
        return cache[t]

    for _, r in feed.iterrows():
        t = r["ticker"]
        rec = pd.Timestamp(r["record_date"])
        reco = str(r.get("board_reco_date", "") or "").strip()
        problems: list[str] = []
        checked += 1

        # 6. value parseable > 0
        try:
            v = float(str(r.get("value", "")).replace(" ", ""))
            if not (v > 0):
                problems.append(f"value not >0 ({r.get('value')!r})")
        except (ValueError, TypeError):
            problems.append(f"value not a float ({r.get('value')!r})")

        # 5. ex == record - 1 TD (weekend-only busday; matches ML anchor sverka check 2)
        try:
            ex = np.datetime64(pd.Timestamp(r["ex_date"]).date())
            if int(np.busday_count(ex, np.datetime64(rec.date()))) != 1:
                problems.append(f"ex_date {r['ex_date']} != record-1TD")
        except Exception:
            problems.append(f"ex_date unparseable ({r.get('ex_date')!r})")

        if not reco:
            problems.append("board_reco_date empty (no-lookahead lead unprovable)")
        else:
            reco_ts = pd.Timestamp(reco)
            df = pub(t)
            # 1. real pubDate of a reco-class disclosure on that day
            same_day = df[(df["pub"].dt.normalize() == reco_ts.normalize())
                          & df["event_name"].astype(str).str.contains(RE_RECO_TITLE)]
            if same_day.empty:
                problems.append(f"board_reco_date {reco} has no matching reco-class disclosure pubDate")
            # 2. <= as_of
            if reco_ts > as_of:
                problems.append(f"board_reco_date {reco} > as_of {as_of.date()} (FUTURE disclosure)")
            # 3. <= record - 12 TD
            deadline = pd.Timestamp(np.busday_offset(rec.date(), -ENTRY_LEAD_TD, roll="backward"))
            if reco_ts.normalize() > deadline:
                problems.append(f"board_reco {reco} later than record-{ENTRY_LEAD_TD}TD "
                                f"({deadline.date()}) — cannot enter in time")

        # 4. authoritative source pubDate <= as_of
        m = re.search(r"EventId=(.+)$", str(r.get("source_url", "")))
        if m:
            df = pub(t)
            srow = df[df["pseudo_guid"] == m.group(1)]
            if not srow.empty and srow["pub"].iloc[0] > as_of:
                problems.append(f"source disclosure pubDate {srow['pub'].iloc[0]} > as_of (FUTURE)")

        if problems:
            viol += 1
            lines.append(f"  [FAIL] {t} record={r['record_date']}: " + "; ".join(problems))

    ok = viol == 0
    stats = {"checked": checked, "violations": viol}
    head = (f"no-lookahead verify: {checked - viol}/{checked} rows PASS  (as_of={as_of.date()}, "
            f"lead={ENTRY_LEAD_TD}TD)")
    return ok, [head] + lines, stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feed", default=str(FEED_CSV))
    ap.add_argument("--as-of", default=date.today().isoformat())
    args = ap.parse_args()

    feed = pd.read_csv(args.feed)
    as_of = pd.Timestamp(args.as_of)
    ok, report, stats = verify(feed, as_of)
    print("\n".join(report))
    print(f"\nVERDICT: {'PASS' if ok else 'FAIL'}  ({stats['violations']} violations / {stats['checked']} rows)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
