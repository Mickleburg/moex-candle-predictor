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
