# -*- coding: utf-8 -*-
r"""Build the FORWARD-looking dividend calendar for the 16-name universe (sleeve S3 run-up).

The ML sleeve enters ~12 trading days before the dividend record (ex) date, so it must know the
UPCOMING record date in advance. MOEX ISS dividends.json only publishes CONFIRMED record dates and
lags (no 2026 dates as of mid-June). The board-recommendation / AGM disclosures on e-disclosure.ru
carry the record date ~37 trading days ahead — that is the lead this feed delivers.

Pipeline:
  1. parquet titles (data/news/edisclosure/) -> dividend-chain candidate events since FETCH_FLOOR.
  2. bodies fetched + cached by llm/scripts/edisc_fetch_bodies.py (event.aspx?EventId=<pseudoGUID>).
  3. parse each body: dividend record date ("...на которую определяются лица, имеющие право на
     получение дивидендов..." + adjacent date), per-(ordinary)-share value, and the declined flag
     ("дивиденды ... не объявлять/не выплачивать"). Declined dividends carry NO record date, which
     cleanly separates payers from decliners.
  4. reconcile per ticker into dividend events keyed by record_date: board_reco_date = earliest
     board/convocation/press-release pubDate that recommended it; agm_date = AGM-decision pubDate;
     status = confirmed (AGM done) | recommended (board only).
  5. keep record_date >= today - 30d (forward + recent); write data/news/dividend_calendar_upcoming.csv.

No-lookahead is intrinsic: every date used is a disclosure pubDate (the same key the historical
certifier uses). Output is informational for the risk/ML layer; is_production stays false.

Run: $env:PYTHONIOENCODING="utf-8"; & "ml\.venv-win\Scripts\python.exe" llm\scripts\build_dividend_calendar_upcoming.py
"""
from __future__ import annotations

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
BODY_DIR = REPO / "data" / "news" / "edisclosure_bodies"
OUT_CSV = REPO / "data" / "news" / "dividend_calendar_upcoming.csv"

TODAY = pd.Timestamp(date.today())
RECENT_FLOOR = TODAY - pd.Timedelta(days=30)   # keep record_date >= today-30d
FETCH_FLOOR = "2026-02-01"
ENTRY_LEAD_TD = 12                              # sleeve enters 12 trading days before record

# Tradeable LINES -> (issuer file key, share class in {ordinary, preferred}). A preferred line shares
# its ordinary issuer's disclosures (one AGM sets ONE record date for both classes) but carries its
# OWN per-share value, so it reads the same bodies and extracts the preferred amount. H9 expansion
# (2026-06-21): 16 originals + prefs SBERP/SNGSP/TATNP + new issuers SIBN/PHOR/MOEX. RTKMP/BSPB are
# provisional on backend's ADTV screen -> not added until confirmed.
_ISSUERS = ("SBER GAZP LKOH GMKN ROSN NVTK TATN MGNT MTSS SNGS CHMF ALRS VTBR MAGN NLMK PLZL "
            "SIBN PHOR MOEX").split()
LINES: dict[str, tuple[str, str]] = {**{t: (t, "ordinary") for t in _ISSUERS},
                                     "SBERP": ("SBER", "preferred"),
                                     "SNGSP": ("SNGS", "preferred"),
                                     "TATNP": ("TATN", "preferred")}
UNIVERSE = list(LINES)   # all tradeable line tickers (issuers + prefs) — build() iterates these

EVENT_URL = "https://www.e-disclosure.ru/portal/event.aspx?EventId="

# ---- title classes (which disclosures can carry a dividend reco/approval) ----
RE_BOARD = re.compile(r"решени[яй] совета директоров|решени.*наблюдательн", re.I)
RE_CONV = re.compile(r"созыв общего собрания|о проведении.*общего собрания акционеров", re.I)
RE_AGM = re.compile(r"решени[яй] общ(?:их|его) собран", re.I)
RE_DIVPR = re.compile(r"рекомендац\w*.{0,40}дивиденд|дивиденд\w*.{0,40}рекомендац", re.I)
CHAIN = re.compile("|".join(x.pattern for x in (RE_BOARD, RE_CONV, RE_AGM, RE_DIVPR)), re.I)

# ---- body parsing ----
MONTHS = {"января": 1, "февраля": 2, "марта": 3, "апреля": 4, "мая": 5, "июня": 6, "июля": 7,
          "августа": 8, "сентября": 9, "октября": 10, "ноября": 11, "декабря": 12}
# the dividend record-date phrase (NOT the AGM-voting "право голоса/участие" date)
RE_REC_PHRASE = re.compile(r"на которую определяются[^.]{0,60}?право на получение дивидендов", re.I)
RE_DECLINE = re.compile(
    r"дивиденд\w*[^.]{0,90}?не\s+(?:объявля|выплач|распределя|начисл)"
    r"|не\s+(?:объявля\w+|выплач\w+|начисл\w+|распределя\w+)[^.]{0,40}дивиденд"
    r"|прибыль[^.]{0,90}?не\s+распределя", re.I)
RE_RK = re.compile(r"(\d+)\s*(?:\([^)]*\)\s*)?руб\w*\.?\s*(\d{1,2})\s*(?:\([^)]*\)\s*)?коп", re.I)
RE_DEC = re.compile(r"(\d+(?:[.,]\d+)?)\s*(?:\([^)]*\)\s*)?руб", re.I)
RE_ORD = re.compile(r"(?:на одну|по)\s+(?:размещ\w+\s+)?обыкновенн\w+\s+(?:именн\w+\s+)?акци\w*", re.I)
RE_PREF = re.compile(r"(?:на одну|по)\s+(?:размещ\w+\s+)?привилегированн\w+\s+(?:именн\w+\s+)?акци\w*", re.I)
RE_GEN = re.compile(r"на одну\s+(?:размещ\w+\s+)?акци\w*", re.I)   # single-class issuers ("на одну акцию")
RE_PAYOUT = re.compile(r"произвести выплату|выплатить дивиденд", re.I)


def parse_date(s: str) -> str | None:
    m = re.search(r"(\d{1,2})\s+([а-яё]+)\s+(\d{4})", s, re.I)
    if m and m.group(2).lower() in MONTHS:
        return f"{int(m.group(3)):04d}-{MONTHS[m.group(2).lower()]:02d}-{int(m.group(1)):02d}"
    m = re.search(r"(\d{2})\.(\d{2})\.(\d{4})", s)
    if m:
        return f"{m.group(3)}-{m.group(2)}-{m.group(1)}"
    return None


def extract_record_date(core: str) -> str | None:
    """The dividend record date is the date adjacent to the 'право на получение дивидендов' phrase.
    The phrase also appears without a date (general references) — those windows yield None."""
    found: list[str] = []
    for m in RE_REC_PHRASE.finditer(core):
        win = core[max(0, m.start() - 80): m.end() + 95]
        d = parse_date(win)
        if d:
            found.append(d)
    if not found:
        return None
    return max(set(found), key=found.count)  # most frequent (one real date per dividend)


def _amount_near(core: str, a_start: int, a_end: int) -> float | None:
    """The per-share amount belonging to a share anchor: the ruble amount CLOSEST to the anchor span,
    whether it sits just AFTER it ('...акцию в размере X руб' / '...акции – X рубля') or just BEFORE
    it ('X руб ... на одну акцию'). Nearest-by-distance is what disambiguates ordinary vs preferred
    when a body lists both on one line (e.g. 'по привилегированной акции – 8,50, по обыкновенной –
    0,90'): each anchor binds to its own adjacent figure. Prefer the explicit 'X руб Y коп' form."""
    lo = max(0, a_start - 80)
    seg = core[lo: a_end + 75]

    def nearest(matches):
        best = None
        for m in matches:
            s, e = lo + m.start(), lo + m.end()
            d = 0 if (s <= a_end and e >= a_start) else min(abs(s - a_end), abs(a_start - e))
            if best is None or d < best[0]:
                best = (d, m)
        return best[1] if best else None

    rk = nearest(list(RE_RK.finditer(seg)))
    if rk is not None:
        return int(rk.group(1)) + int(rk.group(2)) / 100.0
    dec = nearest(list(RE_DEC.finditer(seg)))
    if dec is not None:
        try:
            return float(dec.group(1).replace(" ", "").replace(",", "."))
        except ValueError:
            return None
    return None


def extract_value(core: str, share_class: str = "ordinary") -> tuple[float | None, bool]:
    """Per-share dividend for the given share class. Returns (value, is_total_incl_interim). Prefers
    the amount nearest a 'произвести выплату/выплатить дивиденд' instruction (the sum actually paid at
    record). Ordinary lines also accept a generic 'на одну акцию' (single-class issuers); preferred
    lines require an explicit 'привилегированную акцию' anchor so they never grab the ordinary value."""
    anchors = [(RE_PREF, 0)] if share_class == "preferred" else [(RE_ORD, 0), (RE_GEN, 1)]
    cands: list[tuple[int, float, int]] = []  # (priority, value, pos)
    for rx, prio in anchors:
        for m in rx.finditer(core):
            v = _amount_near(core, m.start(), m.end())
            if v is not None and 0 < v < 100000:
                cands.append((prio, v, m.start()))
    if not cands:
        return None, False
    best_prio = min(c[0] for c in cands)
    pool = [c for c in cands if c[0] == best_prio]
    incl_interim = "в том числе дивиденд" in core.lower()
    payouts = [m.start() for m in RE_PAYOUT.finditer(core)]
    if payouts and len(pool) > 1:
        # prefer the per-share amount closest after a payout instruction
        def dist(c):
            after = [c[2] - p for p in payouts if c[2] >= p]
            return min(after) if after else 10**9
        pool.sort(key=dist)
    return pool[0][1], incl_interim


def classify(name: str) -> str:
    if RE_AGM.search(name):
        return "agm"
    if RE_CONV.search(name):
        return "convoke"
    if RE_DIVPR.search(name):
        return "div_pr"
    return "board"


def body_path(ticker: str, guid: str) -> Path:
    return BODY_DIR / f"{ticker}_{re.sub(r'[^A-Za-z0-9_-]', '_', guid)}.txt"


def td_offset(d: pd.Timestamp, n: int) -> pd.Timestamp:
    return pd.Timestamp(np.busday_offset(d.date(), n, roll="backward"))


def td_gap(a: pd.Timestamp, b: pd.Timestamp) -> int:
    return int(np.busday_count(a.date(), b.date()))


def _issuer_chain(issuer: str, floor: pd.Timestamp,
                  _cache: dict[str, list]) -> list[tuple[pd.Timestamp, str, str, str]]:
    """Dividend-mentioning chain disclosures (pub, cls, guid, body) for an issuer, cached so a
    pref line and its ordinary read the issuer's bodies only once."""
    if issuer in _cache:
        return _cache[issuer]
    out: list[tuple[pd.Timestamp, str, str, str]] = []
    p = DDIR / f"{issuer}.parquet"
    if p.exists():
        d = pd.read_parquet(p, columns=["event_name", "pub_date", "pseudo_guid"])
        d["pub"] = pd.to_datetime(d["pub_date"], format="ISO8601").dt.tz_localize(None).dt.normalize()
        d = d[(d["pub"] >= floor) & d["event_name"].astype(str).str.contains(CHAIN)]
        for _, r in d.iterrows():
            bp = body_path(issuer, r["pseudo_guid"])
            if not bp.exists():
                continue
            core = bp.read_text(encoding="utf-8")
            if "дивиденд" not in core.lower():
                continue
            out.append((r["pub"], classify(str(r["event_name"])), r["pseudo_guid"], core))
    _cache[issuer] = out
    return out


def parse_bodies() -> pd.DataFrame:
    floor = pd.Timestamp(FETCH_FLOOR)
    cache: dict[str, list] = {}
    rows = []
    for line, (issuer, share_class) in LINES.items():
        for pub, cls, guid, core in _issuer_chain(issuer, floor, cache):
            rec = extract_record_date(core)
            val, incl = extract_value(core, share_class)
            declined = bool(RE_DECLINE.search(core)) and rec is None
            rows.append(dict(ticker=line, pub=pub, cls=cls, record=rec, value=val,
                             incl_interim=incl, declined=declined, guid=guid))
    return pd.DataFrame(rows)


def build() -> tuple[pd.DataFrame, pd.DataFrame]:
    msgs = parse_bodies()
    feed_rows, declined_rows = [], []

    for t in UNIVERSE:
        sub = msgs[msgs["ticker"] == t]
        if sub.empty:
            continue
        url = f"{EVENT_URL}"  # filled per-event below
        positive = sub[sub["record"].notna()]
        if positive.empty:
            # no dividend record date disclosed for this name in window
            if sub["declined"].any():
                d = sub[sub["declined"]].sort_values("pub").iloc[-1]
                declined_rows.append(dict(ticker=t, decided=d["pub"].date().isoformat(),
                                          source_url=EVENT_URL + d["guid"]))
            continue

        # group dividend events by record_date
        for rec_date, grp in positive.groupby("record"):
            rec = pd.Timestamp(rec_date)
            board_msgs = grp[grp["cls"].isin(["board", "convoke", "div_pr"])]
            agm_msgs = grp[grp["cls"] == "agm"]
            # board_reco_date = earliest board/convocation/press-release that cites THIS record date
            # (each message's own body carries its record date, so distinct dividend cycles of the
            # same ticker — e.g. PLZL FY2025 final vs. Q1-2026 — stay separated by the groupby).
            board_reco = board_msgs["pub"].min() if not board_msgs.empty else pd.NaT
            agm_date = agm_msgs["pub"].min() if not agm_msgs.empty else pd.NaT
            vals = grp["value"].dropna()
            value = float(vals.iloc[0]) if not vals.empty else None
            incl = bool(grp["incl_interim"].any())
            status = "confirmed" if not agm_msgs.empty else "recommended"
            # authoritative source = AGM decision if present else earliest board reco
            auth = agm_msgs.iloc[0] if not agm_msgs.empty else board_msgs.sort_values("pub").iloc[0] \
                if not board_msgs.empty else grp.sort_values("pub").iloc[0]

            ex = td_offset(rec, -1)
            notes = []
            conf = "high" if status == "confirmed" else "medium"
            if pd.notna(board_reco):
                lead = td_gap(board_reco, rec)
                if lead < ENTRY_LEAD_TD:
                    notes.append(f"board_reco only {lead}TD before record — too late to enter 12TD ahead")
                    conf = "low"
            else:
                notes.append("no board reco pubDate found")
                conf = "low"
            if incl:
                notes.append("interim-netted: value = installment payable at THIS record "
                             "(fiscal-year total is higher; earlier interims already paid)")
            if value is None:
                notes.append("per-share value not parsed")

            feed_rows.append(dict(
                ticker=t,
                record_date=rec.date().isoformat(),
                ex_date=ex.date().isoformat(),
                board_reco_date=board_reco.date().isoformat() if pd.notna(board_reco) else "",
                agm_date=agm_date.date().isoformat() if pd.notna(agm_date) else "",
                value=f"{value:.2f}" if value is not None else "",
                status=status,
                source_url=EVENT_URL + auth["guid"],
                as_of=TODAY.date().isoformat(),
                confidence=conf,
                notes="; ".join(notes),
            ))

    allev = pd.DataFrame(feed_rows, columns=[
        "ticker", "record_date", "ex_date", "board_reco_date", "agm_date",
        "value", "status", "source_url", "as_of", "confidence", "notes"])
    if not allev.empty:
        allev["_rec"] = pd.to_datetime(allev["record_date"])
        allev = allev.sort_values(["_rec", "ticker"]).reset_index(drop=True)
    declined = pd.DataFrame(declined_rows, columns=["ticker", "decided", "source_url"])
    return allev, declined


FEED_COLUMNS = ["ticker", "record_date", "ex_date", "board_reco_date", "agm_date",
                "value", "status", "source_url", "as_of", "confidence", "notes"]


def feed_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (feed, passed, declined). `feed` = forward rows (record_date >= today-30d) to write to
    the CSV; `passed` = extracted-but-already-passed records (record_date < floor); `declined` =
    names that refused a dividend. Shared by main() and the scheduled refresh entry point so both go
    through exactly one build path (deterministic -> idempotent)."""
    allev, declined = build()
    if allev.empty:
        return allev, allev, declined
    feed = allev[allev["_rec"] >= RECENT_FLOOR].drop(columns="_rec").reset_index(drop=True)
    passed = allev[allev["_rec"] < RECENT_FLOOR].drop(columns="_rec").reset_index(drop=True)
    return feed, passed, declined


def main() -> int:
    feed, passed, declined = feed_frames()
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    feed.to_csv(OUT_CSV, index=False, encoding="utf-8")

    print(f"wrote {OUT_CSV}")
    print(f"as_of={TODAY.date()}  (forward window: record_date >= {RECENT_FLOOR.date()})\n")
    if feed.empty:
        print("no upcoming dividend events in window")
    else:
        show = feed[["ticker", "record_date", "ex_date", "board_reco_date", "agm_date",
                     "value", "status", "confidence", "notes"]]
        with pd.option_context("display.max_rows", 60, "display.width", 200,
                               "display.max_colwidth", 60):
            print(show.to_string(index=False))

        # ---- acceptance: board_reco_date <= record_date - 12 TD ----
        f = feed[feed["board_reco_date"] != ""].copy()
        f["rec"] = pd.to_datetime(f["record_date"])
        f["reco"] = pd.to_datetime(f["board_reco_date"])
        f["entry_deadline"] = [td_offset(r, -ENTRY_LEAD_TD) for r in f["rec"]]
        f["slack_days"] = (f["entry_deadline"] - f["reco"]).dt.days
        ok = f[f["slack_days"] >= 0]
        print(f"\n=== ACCEPTANCE: board_reco_date <= record - {ENTRY_LEAD_TD}TD ===")
        print(f"upcoming events in feed: {len(feed)}  (with board_reco: {len(f)})")
        print(f"PASS (enough lead to enter): {len(ok)}/{len(f)}")
        if not f.empty:
            print(f"median slack (board_reco -> record-{ENTRY_LEAD_TD}TD): "
                  f"{f['slack_days'].median():.0f} calendar days  "
                  f"[min {f['slack_days'].min():.0f}, max {f['slack_days'].max():.0f}]")
        late = f[f["slack_days"] < 0]
        if not late.empty:
            print("LATE (cannot enter 12TD ahead):")
            print(late[["ticker", "record_date", "board_reco_date", "slack_days"]].to_string(index=False))

    if not passed.empty:
        print(f"\n=== also extracted: record already passed (< {RECENT_FLOOR.date()}, NOT in feed) ===")
        print(passed[["ticker", "record_date", "board_reco_date", "agm_date",
                      "value", "status"]].to_string(index=False))

    if not declined.empty:
        print(f"\n=== FY2025 dividend DECLINED (no payout — ML should not wait) ===")
        print(declined.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
