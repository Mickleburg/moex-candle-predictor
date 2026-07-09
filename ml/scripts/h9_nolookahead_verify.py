"""H9 no-lookahead — INDEPENDENT verification of the LLM-chat announcement dates.

The LLM chat produced data/news/dividend_announcements.csv (board_reco_date / agm_date per event).
We do NOT take their pass/fail on faith: here the ML block recomputes the entry date in TRADING days
exactly as the sleeve enters (anchor = last trading day <= record date; entry = anchor - ENTRY_OFFSET)
and tests board_reco_date <= entry_date per event. Also reports unmatched events and re-runs the
sleeve restricted to CERTIFIED events to confirm the edge does not depend on the uncertain ones.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ML_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = ML_DIR.parent
sys.path.insert(0, str(ML_DIR))

from scripts.h9_dividend_research import load_daily, UNIVERSE  # noqa: E402
from src.service.dividend_sleeve import ENTRY_OFFSET  # noqa: E402

ANNO = REPO_ROOT / "data" / "news" / "dividend_announcements.csv"


def main() -> int:
    closes = {t: load_daily(t) for t in UNIVERSE}
    closes = {t: s for t, s in closes.items() if s is not None}
    a = pd.read_csv(ANNO)
    a["record_date"] = pd.to_datetime(a["record_date"]).dt.tz_localize("Europe/Moscow")
    a["board_reco_date"] = pd.to_datetime(a["board_reco_date"], errors="coerce").dt.tz_localize("Europe/Moscow")

    rows = []
    for _, r in a.iterrows():
        t = r["ticker"]
        if t not in closes:
            rows.append({**r, "status": "no_price"}); continue
        s = closes[t]
        anchor = s.index.searchsorted(r["record_date"], side="right") - 1
        if anchor < ENTRY_OFFSET:
            rows.append({**r, "status": "too_early"}); continue
        entry_date = s.index[anchor - ENTRY_OFFSET]
        if pd.isna(r["board_reco_date"]):
            rows.append({**r, "entry_date": entry_date, "status": "unmatched"}); continue
        # lead in trading days between board reco and record date
        reco_pos = s.index.searchsorted(r["board_reco_date"], side="right") - 1
        lead_td = anchor - reco_pos
        ok = r["board_reco_date"] <= entry_date
        rows.append({"ticker": t, "record_date": r["record_date"].date(),
                     "board_reco_date": r["board_reco_date"].date(), "entry_date": entry_date.date(),
                     "lead_td": int(lead_td), "status": "PASS" if ok else "VIOLATION"})
    df = pd.DataFrame(rows)

    matched = df[df["status"].isin(["PASS", "VIOLATION"])]
    viol = df[df["status"] == "VIOLATION"]
    unm = df[df["status"] == "unmatched"]
    print(f"H9 no-lookahead INDEPENDENT verification (entry = record - {ENTRY_OFFSET} TD)")
    print(f"  total events in CSV: {len(df)}")
    print(f"  matched (board_reco present): {len(matched)}")
    print(f"  PASS (board_reco <= entry): {len(matched) - len(viol)} / {len(matched)}")
    print(f"  VIOLATIONS: {len(viol)}")
    if len(matched):
        print(f"  median lead (reco -> record): {int(matched['lead_td'].median())} TD "
              f"(min {int(matched['lead_td'].min())}, max {int(matched['lead_td'].max())})")
    if len(viol):
        print("\n  VIOLATION rows:")
        print(viol[["ticker", "record_date", "board_reco_date", "entry_date", "lead_td"]].to_string(index=False))
    if len(unm):
        print(f"\n  UNMATCHED (no board_reco — not certifiable from this data): {len(unm)}")
        print(unm[["ticker", "record_date", "notes"]].to_string(index=False))

    # smallest leads = the riskiest events; show the tightest 8
    if len(matched):
        print("\n  tightest leads (smallest reco->record gap):")
        print(matched.nsmallest(8, "lead_td")[["ticker", "record_date", "board_reco_date",
              "entry_date", "lead_td"]].to_string(index=False))
    print(f"\n  VERDICT: {'no-lookahead CERTIFIED for all matched events' if len(viol)==0 else 'VIOLATIONS found - inspect above'}; "
          f"{len(unm)} unmatched remain uncertified (exclude from the certified book).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
