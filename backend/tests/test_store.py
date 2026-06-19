"""Store tests: idempotent merge + single-file consolidation."""

import pandas as pd

from backend import store
from backend.tests.conftest import make_candles


def test_merge_increment_dedups_and_sorts(trading_days):
    existing = make_candles("SBER", "1D", trading_days[:5])
    # overlap last day + 3 new days, with the overlapping bar mutated (re-fetch wins)
    fresh = make_candles("SBER", "1D", trading_days[4:8], base=200.0)
    merged = store.merge_increment(existing, fresh)
    assert merged["begin"].is_monotonic_increasing
    assert merged["begin"].is_unique
    assert len(merged) == 8                       # 5 + 4 - 1 overlap
    # the re-fetched overlapping bar overwrote the old one (keep="last")
    overlap_begin = make_candles("SBER", "1D", trading_days[4:5], base=200.0)["begin"].iloc[0]
    row = merged.loc[merged["begin"] == overlap_begin]
    assert row["open"].iloc[0] == 200.0


def test_merge_increment_idempotent_on_same_data(trading_days):
    existing = make_candles("SBER", "1D", trading_days[:5])
    once = store.merge_increment(existing, existing)
    twice = store.merge_increment(once, existing)
    pd.testing.assert_frame_equal(once, twice)
    assert len(once) == 5


def test_write_consolidated_single_file_and_idempotent(tmp_path, trading_days):
    df = make_candles("SBER", "1D", trading_days[:6])
    p1 = store.write_consolidated(df, "SBER", "1D", data_dir=tmp_path)
    assert p1.exists()
    assert len(store.store_files("SBER", "1D", tmp_path)) == 1

    # re-run on identical data: still one file, identical contents, no growth
    reloaded = store.load_ticker("SBER", "1D", tmp_path)
    merged = store.merge_increment(reloaded, df)
    p2 = store.write_consolidated(merged, "SBER", "1D", data_dir=tmp_path)
    assert len(store.store_files("SBER", "1D", tmp_path)) == 1
    again = store.load_ticker("SBER", "1D", tmp_path)
    pd.testing.assert_frame_equal(
        reloaded.reset_index(drop=True), again.reset_index(drop=True))


def test_write_consolidated_removes_stale_named_file(tmp_path, trading_days):
    df = make_candles("SBER", "1D", trading_days[:4])
    store.write_consolidated(df, "SBER", "1D", data_dir=tmp_path)
    # extend by two days -> filename range changes; old file must be removed
    bigger = store.merge_increment(df, make_candles("SBER", "1D", trading_days[3:6]))
    store.write_consolidated(bigger, "SBER", "1D", data_dir=tmp_path)
    files = store.store_files("SBER", "1D", tmp_path)
    assert len(files) == 1
    assert store.last_begin("SBER", "1D", tmp_path) == pd.Timestamp(bigger["begin"].max())
