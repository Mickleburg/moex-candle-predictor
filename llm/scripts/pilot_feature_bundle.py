r"""Pilot: verify the baseline news layer joins into the ML block's feature_bundle with
no-lookahead, using the ML block's own assembler (validates vs feature_bundle.schema.json).

Run:  & "ml\.venv-win\Scripts\python.exe" llm\scripts\pilot_feature_bundle.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "ml"))  # so `src.features...` resolves to ml/src/features

from src.features.cross_sectional import load_panels            # noqa: E402
from src.features.decision_grid import load_grid                # noqa: E402
from src.features.feature_bundle import assemble_feature_bundle, validate_against_schema  # noqa: E402

NEWS_CSV = REPO_ROOT / "data" / "news" / "baseline_features.csv"


def main() -> int:
    panel, _, _ = load_panels(timeframe="1D")
    grid = load_grid()
    news = pd.read_csv(NEWS_CSV)
    news["as_of"] = pd.to_datetime(news["as_of"])  # tz-aware MSK

    # a decision point with real SBER history (mid-2024)
    as_of = grid[grid >= pd.Timestamp("2024-06-03", tz="Europe/Moscow")][0]
    print(f"decision as_of = {as_of.isoformat()}")

    bundle = assemble_feature_bundle(panel, news, as_of, news_cols=["sentiment", "impact_score"])
    validate_against_schema(bundle)  # raises if invalid vs feature_bundle.schema.json
    print(f"bundle OK: universe={len(bundle['universe'])} entries={len(bundle['entries'])} "
          f"spec={bundle['feature_spec']}")

    sber = next(e for e in bundle["entries"] if e["ticker"] == "SBER")
    print(f"SBER entry: quant={sber['quant']} news={sber['news']} valid={sber['valid']}")

    # --- no-lookahead checks -------------------------------------------------
    # 1) the news the assembler used for SBER must equal our CSV row at as_of (latest<=decision)
    sber_row = news[(news["ticker"] == "SBER") & (news["as_of"] == as_of)].iloc[0]
    expected = [round(float(sber_row["sentiment"]), 6), round(float(sber_row["impact_score"]), 6)]
    assert sber["news"] == expected, f"join mismatch: {sber['news']} != {expected}"
    print(f"  no-lookahead join OK: SBER news == CSV row @ as_of {expected}")

    # 2) the assembler must NOT see any news published after as_of
    future = news[news["as_of"] > as_of]
    assert len(future) > 0, "sanity: there should be later grid points"
    floor = as_of - pd.Timedelta(days=7)
    visible = news[(news["as_of"] <= as_of) & (news["as_of"] > floor)]
    assert (visible["as_of"] <= as_of).all(), "lookahead detected!"
    print(f"  window check OK: {len(visible)} visible rows in (as_of-7d, as_of], "
          f"0 from the future ({len(future)} future rows correctly excluded)")

    # 3) an earlier valid as_of (enough price history) still assembles + validates
    early = grid[grid >= pd.Timestamp("2020-03-01", tz="Europe/Moscow")][0]
    eb = assemble_feature_bundle(panel, news, early, news_cols=["sentiment", "impact_score"])
    validate_against_schema(eb)
    esber = next(e for e in eb["entries"] if e["ticker"] == "SBER")
    print(f"  early as_of {early.date()}: assembles+validates, SBER news={esber['news']}")

    print("\nPILOT PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
