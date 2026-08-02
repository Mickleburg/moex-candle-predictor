"""Dividend-backfill tests: existing-wins merge, provenance tagging, corroboration flag."""

import json

import pandas as pd

from backend import dividends


def _existing_csv(tmp_path):
    p = tmp_path / "dividends.csv"
    pd.DataFrame({
        "ticker": ["SBER", "SBER"],
        "date": ["2024-07-11", "2025-07-18"],
        "value": [33.3, 34.84],
        "ccy": ["RUB", "RUB"],
    }).to_csv(p, index=False)
    return p


def _fake_fetch(table):
    def fetch(tk):
        rows = table.get(tk, [])
        return pd.DataFrame(rows, columns=["ticker", "date", "value", "ccy"])
    return fetch


def test_adds_new_lines_and_tags_source(tmp_path):
    csv = _existing_csv(tmp_path)
    prov = tmp_path / "prov.json"
    fetch = _fake_fetch({
        "SBER": [("SBER", "2024-07-11", 33.3, "RUB"), ("SBER", "2025-07-18", 34.84, "RUB")],
        "SBERP": [("SBERP", "2025-07-18", 34.84, "RUB"), ("SBERP", "2024-07-11", 33.3, "RUB")],
    })
    rep = dividends.backfill(("SBER", "SBERP"), csv, fetch_fn=fetch,
                             run_date="2026-06-26", provenance_path=prov)
    out = pd.read_csv(csv)
    # existing 2 SBER rows kept as iss_history; both SBERP rows added & tagged
    assert rep["rows_added_this_run"] == 2
    sberp = out[out["ticker"] == "SBERP"]
    assert set(sberp["source"]) == {"iss_backfill_2026-06-26"}
    assert set(out[out["ticker"] == "SBER"]["source"]) == {"iss_history"}


def test_existing_value_wins_discrepancy_reported_not_applied(tmp_path):
    csv = _existing_csv(tmp_path)
    prov = tmp_path / "prov.json"
    # ISS returns a DIFFERENT value for an existing (SBER, 2025-07-18)
    fetch = _fake_fetch({"SBER": [("SBER", "2025-07-18", 99.99, "RUB")]})
    rep = dividends.backfill(("SBER",), csv, fetch_fn=fetch,
                             run_date="2026-06-26", provenance_path=prov)
    out = pd.read_csv(csv)
    kept = out[(out["ticker"] == "SBER") & (out["date"] == "2025-07-18")]["value"].iloc[0]
    assert kept == 34.84                                   # stored value preserved
    assert rep["rows_added_this_run"] == 0
    disc = rep["value_discrepancies_reported_not_applied"]
    assert len(disc) == 1 and disc[0]["stored"] == 34.84 and disc[0]["iss"] == 99.99


def test_corroboration_window_flagged(tmp_path):
    csv = _existing_csv(tmp_path)
    prov = tmp_path / "prov.json"
    # a NEW record inside the burned-split overlap window must be flagged, not silently mixed
    fetch = _fake_fetch({"TATN": [("TATN", "2025-10-08", 38.2, "RUB"),
                                  ("TATN", "2024-07-09", 25.17, "RUB")]})
    rep = dividends.backfill(("TATN",), csv, fetch_fn=fetch,
                             run_date="2026-06-26", provenance_path=prov)
    cw = rep["corroboration_window"]
    assert cw["rows_added_in_window"] == 1
    assert cw["tickers_in_window"] == ["TATN"]
    assert json.loads(prov.read_text())["corroboration_window"]["since"] == "2025-08-01"


def test_idempotent_second_run_adds_nothing(tmp_path):
    csv = _existing_csv(tmp_path)
    prov = tmp_path / "prov.json"
    fetch = _fake_fetch({"SBERP": [("SBERP", "2025-07-18", 34.84, "RUB")]})
    dividends.backfill(("SBERP",), csv, fetch_fn=fetch, run_date="2026-06-26", provenance_path=prov)
    rep2 = dividends.backfill(("SBERP",), csv, fetch_fn=fetch, run_date="2026-06-27", provenance_path=prov)
    assert rep2["rows_added_this_run"] == 0


# --- promote_events: e-disclosure realized events -> permanent history (stops the rolling-window loss)


def test_promote_adds_realized_events_tagged_source(tmp_path):
    csv = _existing_csv(tmp_path)
    prov = tmp_path / "prov.json"
    ev = pd.DataFrame({"ticker": ["SNGS", "MOEX"], "date": ["2026-07-16", "2026-07-09"],
                       "value": [4.73, 26.11]})
    rep = dividends.promote_events(ev, source="e-disclosure", csv_path=csv,
                                   run_date="2026-09-01", provenance_path=prov)
    out = pd.read_csv(csv)
    assert rep["rows_added_this_run"] == 2
    assert set(out[out["ticker"].isin(["SNGS", "MOEX"])]["source"]) == {"e-disclosure"}
    assert set(out[out["ticker"] == "SBER"]["source"]) == {"iss_history"}   # history untouched


def test_promote_existing_value_wins_discrepancy_reported(tmp_path):
    csv = _existing_csv(tmp_path)
    prov = tmp_path / "prov.json"
    # e-disclosure parses a DIFFERENT value for an event ISS already has -> keep stored, report clash
    ev = pd.DataFrame({"ticker": ["SBER"], "date": ["2025-07-18"], "value": [99.99]})
    rep = dividends.promote_events(ev, csv_path=csv, run_date="2026-09-01", provenance_path=prov)
    out = pd.read_csv(csv)
    kept = out[(out["ticker"] == "SBER") & (out["date"] == "2025-07-18")]["value"].iloc[0]
    assert kept == 34.84 and rep["rows_added_this_run"] == 0
    disc = rep["value_discrepancies_reported_not_applied"]
    assert len(disc) == 1 and disc[0]["stored"] == 34.84 and disc[0]["incoming"] == 99.99


def test_promote_idempotent(tmp_path):
    csv = _existing_csv(tmp_path)
    prov = tmp_path / "prov.json"
    ev = pd.DataFrame({"ticker": ["SNGS"], "date": ["2026-07-16"], "value": [4.73]})
    dividends.promote_events(ev, csv_path=csv, run_date="2026-09-01", provenance_path=prov)
    csv_after, prov_after = csv.read_text(), prov.read_text()

    rep2 = dividends.promote_events(ev, csv_path=csv, run_date="2026-09-02", provenance_path=prov)
    assert rep2["rows_added_this_run"] == 0             # re-promoting the same event is a no-op
    # ...and a TRUE no-op: neither file is rewritten. A bumped `generated_at` made an otherwise
    # byte-identical refresh show up as a repo change.
    assert csv.read_text() == csv_after
    assert prov.read_text() == prov_after


def test_promote_filters_nonpositive_and_normalizes_date(tmp_path):
    csv = _existing_csv(tmp_path)
    prov = tmp_path / "prov.json"
    ev = pd.DataFrame({"ticker": ["AAA", "BBB"],
                       "date": [pd.Timestamp("2026-07-16"), "2026-07-17"],
                       "value": [0.0, 5.0]})            # AAA value 0 -> dropped (ML loader would too)
    rep = dividends.promote_events(ev, csv_path=csv, run_date="2026-09-01", provenance_path=prov)
    out = pd.read_csv(csv)
    assert rep["rows_added_this_run"] == 1 and "AAA" not in set(out["ticker"])
    assert out[out["ticker"] == "BBB"]["date"].iloc[0] == "2026-07-17"   # Timestamp -> YYYY-MM-DD


def test_promote_empty_is_noop(tmp_path):
    csv = _existing_csv(tmp_path)
    before = csv.read_text()
    rep = dividends.promote_events(pd.DataFrame(columns=["ticker", "date", "value"]),
                                   csv_path=csv, provenance_path=tmp_path / "prov.json")
    assert rep["rows_added_this_run"] == 0 and csv.read_text() == before   # file untouched
