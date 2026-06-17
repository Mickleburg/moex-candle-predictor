"""Correctness tests for the V2 news-feature block (run: python -m pytest llm/test_news_features.py).

Covers the invariants the ML block requires before trusting the news layer:
  * no-lookahead strictly by PUBLICATION time (pub_date), never event_date;
  * trailing window honoured;
  * source disclosures deduplicated (unique pseudo_guid per ticker);
  * news<->ticker mapping pinned (one e-disclosure company_id per ticker file);
  * timezone handling: as_of normalised to MSK like the decision grid.
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "llm" / "src"))

from llm_ta import features as feat  # noqa: E402

DISCLOSURES_DIR = REPO_ROOT / "data" / "news" / "edisclosure"
MSK = feat.MSK

# pinned ticker -> e-disclosure company_id (see llm/docs/NEWS_SOURCE_EDISCLOSURE.md)
EXPECTED_COMPANY_ID = {
    "SBER": 3043, "GAZP": 934, "LKOH": 17, "GMKN": 564, "ROSN": 6505, "NVTK": 225,
    "TATN": 118, "MGNT": 7671, "MTSS": 236, "SNGS": 312, "CHMF": 30, "ALRS": 199,
    # extended dividend-certification universe (discovered 2026-06-17)
    "VTBR": 1210, "MAGN": 9, "NLMK": 2509, "PLZL": 7832,
}
ALL_TICKERS = list(EXPECTED_COMPANY_ID)


def _d(pub_iso: str, name: str = "событие", etype: str = "other") -> feat.Disclosure:
    return feat.Disclosure(pub_date=dt.datetime.fromisoformat(pub_iso), event_name=name,
                           event_type=etype, pseudo_guid="g", agency="ИНТЕРФАКС")


# --- no-lookahead by PUBLICATION time --------------------------------------------------

def test_no_lookahead_excludes_future_publication():
    as_of = dt.datetime(2024, 6, 3, tzinfo=MSK)
    items = [
        _d("2024-06-01T10:00:00+03:00"),          # in window, published before as_of -> in
        _d("2024-06-03T00:00:01+03:00"),          # published 1s AFTER as_of -> excluded
        _d("2024-06-10T10:00:00+03:00"),          # well after as_of -> excluded
    ]
    feats, window = feat.compute_features(items, as_of, window_hours=168)
    assert feats["news_count"] == 1
    assert all(d.pub_date <= as_of for d in window)


def test_no_lookahead_uses_pub_date_not_event_date():
    # The pipeline must key on publication, not event occurrence. Disclosure carries no
    # event_date at all -> a release published after as_of cannot leak even if it describes
    # an earlier event. (Guards against ever joining on event_date.)
    assert "event_date" not in feat.Disclosure.__slots__ if hasattr(feat.Disclosure, "__slots__") \
        else "event_date" not in feat.Disclosure.__dataclass_fields__
    as_of = dt.datetime(2024, 6, 3, tzinfo=MSK)
    published_after = [_d("2024-06-04T09:00:00+03:00", "Дивиденды (событие было 2024-06-01)")]
    feats, window = feat.compute_features(published_after, as_of, window_hours=168)
    assert feats["news_count"] == 0 and window == []


def test_trailing_window_boundary():
    as_of = dt.datetime(2024, 6, 3, tzinfo=MSK)
    # window_start = as_of - 168h = 2024-05-27T00:00 MSK
    items = [
        _d("2024-05-26T23:00:00+03:00"),          # before window_start -> excluded
        _d("2024-05-28T00:00:00+03:00"),          # within 168h -> in
        _d("2024-06-02T23:00:00+03:00"),          # in
    ]
    feats, _ = feat.compute_features(items, as_of, window_hours=168)
    assert feats["news_count"] == 2


def test_recency_and_empty_window():
    as_of = dt.datetime(2024, 6, 3, tzinfo=MSK)
    feats, window = feat.compute_features([], as_of, window_hours=168)
    assert feats["news_count"] == 0 and feats["event_type"] == "none" and window == []
    feats2, _ = feat.compute_features([_d("2024-06-02T18:00:00+03:00")], as_of, window_hours=168)
    assert feats2["recency_minutes"] == pytest.approx(6 * 60, abs=1)  # 18:00 -> 00:00 = 6h


# --- timezone handling ----------------------------------------------------------------

def test_to_msk_naive_assumed_moscow():
    naive = dt.datetime(2024, 6, 3, 12, 0, 0)
    out = feat.to_msk(naive)
    assert out.tzinfo is not None and out.utcoffset() == dt.timedelta(hours=3)
    assert (out.hour, out.minute) == (12, 0)


def test_to_msk_utc_converted():
    utc = dt.datetime(2024, 6, 2, 21, 0, 0, tzinfo=dt.timezone.utc)  # = 00:00 MSK next day
    out = feat.to_msk(utc)
    assert out.utcoffset() == dt.timedelta(hours=3)
    assert (out.year, out.month, out.day, out.hour) == (2024, 6, 3, 0)


def test_build_analysis_emits_tz_aware_msk():
    items = [_d("2024-06-02T18:00:00+03:00")]
    a = feat.build_analysis("SBER", dt.datetime(2024, 6, 3, tzinfo=MSK),
                            window_hours=168, disclosures=items)
    assert a["as_of"].endswith("+03:00")
    for s in a["sources"]:
        assert dt.datetime.fromisoformat(s["published_at"]) <= dt.datetime.fromisoformat(a["as_of"])
    assert a["is_production"] is False


# --- source data integrity: dedup + ticker mapping ------------------------------------

@pytest.mark.parametrize("ticker", ALL_TICKERS)
def test_no_duplicate_disclosures(ticker):
    path = DISCLOSURES_DIR / f"{ticker}.parquet"
    if not path.exists():
        pytest.skip(f"{path} not present")
    df = pd.read_parquet(path, columns=["pseudo_guid"])
    assert df["pseudo_guid"].duplicated().sum() == 0, f"{ticker}: duplicate disclosures"


@pytest.mark.parametrize("ticker", ALL_TICKERS)
def test_ticker_company_mapping(ticker):
    path = DISCLOSURES_DIR / f"{ticker}.parquet"
    if not path.exists():
        pytest.skip(f"{path} not present")
    df = pd.read_parquet(path, columns=["company_id"])
    ids = set(int(x) for x in df["company_id"].unique())
    assert ids == {EXPECTED_COMPANY_ID[ticker]}, \
        f"{ticker}: expected company_id {EXPECTED_COMPANY_ID[ticker]}, got {ids}"


# --- dividend-announcement certification table (H9 run-up no-lookahead) -----------------

DIV_ANN = REPO_ROOT / "data" / "news" / "dividend_announcements.csv"
DIV_RAW = REPO_ROOT / "data" / "raw" / "dividends.csv"


def _load_div_ann():
    if not DIV_ANN.exists():
        pytest.skip(f"{DIV_ANN} not present (run build_dividend_announcements.py)")
    return pd.read_csv(DIV_ANN, dtype=str)


def test_div_ann_schema_and_dates():
    df = _load_div_ann()
    assert list(df.columns) == ["ticker", "record_date", "board_reco_date",
                                "agm_date", "source_url", "confidence", "notes"]
    assert df["confidence"].isin({"high", "medium", "low", "none"}).all()
    # every populated date is ISO and board_reco/agm precede the record date
    for _, r in df.iterrows():
        rec = dt.date.fromisoformat(r["record_date"])
        for col in ("board_reco_date", "agm_date"):
            if isinstance(r[col], str) and r[col]:
                d = dt.date.fromisoformat(r[col])  # raises if not ISO
                assert d < rec, f"{r['ticker']} {r['record_date']}: {col} {d} not before record"


def test_div_ann_join_integrity():
    # every (ticker, record_date) must exist verbatim in dividends.csv for the ML join
    ann = _load_div_ann()
    raw = pd.read_csv(DIV_RAW, dtype=str)
    raw_keys = set(zip(raw["ticker"], raw["date"]))
    for _, r in ann.iterrows():
        assert (r["ticker"], r["record_date"]) in raw_keys, \
            f"{r['ticker']} {r['record_date']} not in dividends.csv"


def test_div_ann_no_lookahead_12td():
    # the certification claim: every MATCHED event was publicly announced (board reco) at least
    # 12 trading days before its record date. Unmatched (board_reco empty) are out of scope here.
    import numpy as np
    df = _load_div_ann()
    matched = df[df["board_reco_date"].astype(str).str.len() > 0]
    late = []
    for _, r in matched.iterrows():
        reco = dt.date.fromisoformat(r["board_reco_date"])
        rec = dt.date.fromisoformat(r["record_date"])
        if int(np.busday_count(reco, rec)) < 12:
            late.append((r["ticker"], r["record_date"], r["board_reco_date"]))
    assert not late, f"events announced within 12 TD of record (lookahead risk): {late}"
