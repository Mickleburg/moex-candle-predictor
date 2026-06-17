# -*- coding: utf-8 -*-
r"""Certify the dividend pre-ex run-up edge as no-lookahead: for every dividend event in
data/raw/dividends.csv, find the FIRST public disclosure of the dividend on e-disclosure.ru.

In Russian law a dividend is recommended by the board (совет директоров), which at the same
meeting convenes the AGM (созыв общего собрания) with the dividend on the agenda; the AGM then
approves it (решения общего собрания), and a record date is set ~10-20 days later. So the chain
is:  board reco / AGM convocation  ->  AGM decision  ->  record date.

We mine the e-disclosure substantial-fact TITLES (the dividend amount lives in the body, which we
don't store, but the title classes pin the chain unambiguously):
  * board_reco_date = earliest public disclosure of the recommending board meeting = the earlier of
      the "Решения совета директоров" and the "Созыв общего собрания акционеров" that pair together
      (same board meeting). This is the no-lookahead-relevant date.
  * agm_date        = "Решения общих собраний акционеров" (AGM approval).

No-lookahead: matching uses pubDate (publication time) only; eventDate is never used.

Output: data/news/dividend_announcements.csv
  ticker, record_date, board_reco_date, agm_date, source_url, confidence, notes

Run: $env:PYTHONIOENCODING="utf-8"; & "ml\.venv-win\Scripts\python.exe" llm\scripts\build_dividend_announcements.py
"""
from __future__ import annotations

import re
import sys
import io
from pathlib import Path

import numpy as np
import pandas as pd

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

REPO = Path(__file__).resolve().parents[2]
DDIR = REPO / "data" / "news" / "edisclosure"
DIV_CSV = REPO / "data" / "raw" / "dividends.csv"
OUT_CSV = REPO / "data" / "news" / "dividend_announcements.csv"

# ticker -> e-disclosure companyID (for the source_url; pins the issuer)
COMPANY_ID = {
    "SBER": 3043, "GAZP": 934, "LKOH": 17, "GMKN": 564, "ROSN": 6505, "NVTK": 225,
    "TATN": 118, "MGNT": 7671, "MTSS": 236, "SNGS": 312, "CHMF": 30, "ALRS": 199,
    "VTBR": 1210, "MAGN": 9, "NLMK": 2509, "PLZL": 7832,
}
UNIVERSE = list(COMPANY_ID)

# substantial-fact title classes
RE_AGM_DECISION = re.compile(
    r"решени[яй].*общ(?:их|его) собран|общим собранием.*приня|собранием акционеров.*приня"
    r"|состоялось.*собрание акционеров|результат.*общего собрания акционеров", re.I)
RE_AGM_CONVOKE = re.compile(
    r"созыв общего собрания|о проведении.*общего собрания акционеров"
    r"|проведени[еия].*общего собрания акционеров", re.I)
RE_BOARD = re.compile(
    r"решени[яй] совета директоров|решени.*наблюдательн"
    r"|результат.*заседани.*совета директоров|ключев.*решени.*совет.*директоров", re.I)
# explicit dividend-recommendation press release (gold standard when present, e.g. PLZL/ROSN)
RE_DIV_RECO = re.compile(
    r"рекомендац\w*.{0,40}дивиденд|дивиденд\w*.{0,40}рекомендац"
    r"|рекомендова\w*.{0,40}дивиденд|дивиденд\w*.{0,40}рекомендова", re.I)
AGM_NOTICE_MIN_DAYS = 18  # legal AGM-notice floor: convocation must precede the AGM by >= this

DATA_FLOOR = pd.Timestamp("2020-01-01")  # disclosure history begins here


def load(ticker: str) -> pd.DataFrame | None:
    p = DDIR / f"{ticker}.parquet"
    if not p.exists():
        return None
    d = pd.read_parquet(p, columns=["event_name", "pub_date", "pseudo_guid"])
    d["pub"] = pd.to_datetime(d["pub_date"], format="ISO8601").dt.tz_localize(None).dt.normalize()
    d["name"] = d["event_name"].astype(str)
    return d.sort_values("pub").reset_index(drop=True)


def nearest_before(df, regex, upper, lower_days):
    lo = upper - pd.Timedelta(days=lower_days)
    sub = df[(df["pub"] < upper) & (df["pub"] >= lo) & df["name"].str.contains(regex, regex=True)]
    return None if sub.empty else sub.iloc[-1]  # latest (df sorted by pub)


def board_paired_with(df, convoke_pub):
    """The 'Решения СД' that produced this convocation: within [-7d, +1d] of the convoke."""
    lo = convoke_pub - pd.Timedelta(days=7)
    hi = convoke_pub + pd.Timedelta(days=1)
    sub = df[(df["pub"] >= lo) & (df["pub"] <= hi) & df["name"].str.contains(RE_BOARD, regex=True)]
    return None if sub.empty else sub.iloc[-1]


def td_gap(start: pd.Timestamp, end: pd.Timestamp) -> int:
    """Trading-day count between two dates (weekdays only; MOEX holidays ignored — approx)."""
    return int(np.busday_count(start.date(), end.date()))


def main() -> int:
    div = pd.read_csv(DIV_CSV)
    div["record"] = pd.to_datetime(div["date"])
    div = div[(div["record"] >= DATA_FLOOR) & (div["record"] <= "2025-12-31")].copy()
    div = div.sort_values(["ticker", "record"]).reset_index(drop=True)

    out = []
    for t in UNIVERSE:
        df = load(t)
        cid = COMPANY_ID[t]
        url = f"https://www.e-disclosure.ru/portal/company.aspx?id={cid}"
        for _, r in div[div["ticker"] == t].iterrows():
            rec = r["record"]
            if df is None:
                out.append(dict(ticker=t, record_date=rec.date().isoformat(),
                                board_reco_date="", agm_date="", source_url=url,
                                confidence="none", notes="no disclosure data for ticker"))
                continue

            agm = nearest_before(df, RE_AGM_DECISION, rec, 75)
            # the convocation belongs to THIS AGM: it must precede the AGM decision by the legal
            # notice gap (>= AGM_NOTICE_MIN_DAYS), else it is a convocation for a different meeting.
            conv_upper = (agm["pub"] - pd.Timedelta(days=AGM_NOTICE_MIN_DAYS)) if agm is not None \
                else rec - pd.Timedelta(days=AGM_NOTICE_MIN_DAYS)
            convoke = nearest_before(df, RE_AGM_CONVOKE, conv_upper + pd.Timedelta(days=1), 90)

            board = None
            if convoke is not None:
                board = board_paired_with(df, convoke["pub"])
            if board is None:  # no convocation found, or no paired board fact -> nearest board fact
                board = nearest_before(df, RE_BOARD, conv_upper + pd.Timedelta(days=1), 75)

            # explicit dividend-recommendation press release (strongest signal when present)
            divpr = nearest_before(df, RE_DIV_RECO, rec, 90)

            # board_reco_date = earliest public disclosure of the recommendation for this dividend
            cands = [x["pub"] for x in (board, convoke, divpr) if x is not None]
            reco = min(cands) if cands else None

            notes = []
            if divpr is not None:
                notes.append(f"div_reco_pr {divpr['pub'].date()}")
            if convoke is not None:
                notes.append(f"convoke {convoke['pub'].date()}")
            if board is not None:
                notes.append(f"board_sf {board['pub'].date()}")
            if agm is not None:
                notes.append(f"agm {agm['pub'].date()}")

            # confidence
            if reco is None:
                conf = "none"
                if rec - DATA_FLOOR <= pd.Timedelta(days=90):
                    notes.append("reco predates 2020-01-01 disclosure floor (record in early 2020)")
                else:
                    notes.append("no board/AGM disclosure matched")
            else:
                gap_td = td_gap(reco, rec)
                ordered = (agm is None) or (reco <= agm["pub"] < rec)
                strong = (divpr is not None) or (convoke is not None and agm is not None)
                if gap_td < 12:
                    conf = "low"
                    notes.append(f"reco within {gap_td}TD of record — lookahead risk")
                elif ordered and strong:
                    conf = "high"
                else:
                    conf = "medium"
                    if not ordered:
                        notes.append("chain order irregular")

            out.append(dict(
                ticker=t, record_date=rec.date().isoformat(),
                board_reco_date=reco.date().isoformat() if reco is not None else "",
                agm_date=agm["pub"].date().isoformat() if agm is not None else "",
                source_url=url, confidence=conf, notes="; ".join(notes)))

    res = pd.DataFrame(out, columns=["ticker", "record_date", "board_reco_date",
                                     "agm_date", "source_url", "confidence", "notes"])
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT_CSV, index=False, encoding="utf-8")

    # ---- report ----
    n = len(res)
    matched = res[res["board_reco_date"] != ""].copy()
    matched["rec_dt"] = pd.to_datetime(matched["record_date"])
    matched["reco_dt"] = pd.to_datetime(matched["board_reco_date"])
    matched["gap_cal"] = (matched["rec_dt"] - matched["reco_dt"]).dt.days
    matched["gap_td"] = [td_gap(a, b) for a, b in zip(matched["reco_dt"], matched["rec_dt"])]
    late = matched[matched["gap_td"] < 12]

    print(f"wrote {OUT_CSV}  ({n} events, 16 tickers, record_date 2020-2025)")
    print(f"\ncoverage: board_reco found {len(matched)}/{n} ({len(matched)/n:.0%})"
          f" | agm found {(res['agm_date'] != '').sum()}/{n}")
    print("confidence:", res["confidence"].value_counts().to_dict())
    print(f"\nreco->record gap (calendar days): median {matched['gap_cal'].median():.0f}"
          f"  min {matched['gap_cal'].min():.0f}  max {matched['gap_cal'].max():.0f}")
    print(f"reco->record gap (trading days):  median {matched['gap_td'].median():.0f}"
          f"  min {matched['gap_td'].min():.0f}  max {matched['gap_td'].max():.0f}")
    print(f"\nNO-LOOKAHEAD TEST  board_reco_date <= record - 12 TD:")
    print(f"  PASS: {len(matched) - len(late)}/{len(matched)}  |  events announced LATER than"
          f" record-12TD: {len(late)}")
    if len(late):
        print(late[["ticker", "record_date", "board_reco_date", "gap_td", "confidence"]].to_string(index=False))
    miss = res[res["board_reco_date"] == ""]
    if len(miss):
        print(f"\nunmatched ({len(miss)}):")
        print(miss[["ticker", "record_date", "confidence", "notes"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
