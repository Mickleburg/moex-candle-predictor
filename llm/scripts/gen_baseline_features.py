r"""Generate the BASELINE news-feature layer on the shared decision grid (free, no LLM).

For every (ticker, as_of) in the decision grid (data/features/decision_grid.csv) we compute the
deterministic news features over a trailing 7-day window with strict no-lookahead (pub_date<=as_of),
and write a tidy table for the ML block's assemble_feature_bundle / h2_fusion_sim.

Output: data/news/baseline_features.csv
  columns: ticker, as_of (ISO tz-aware MSK), sentiment, impact_score, novelty,
           event_type, news_count, recency_minutes

Run:  $env:PYTHONPATH="llm\src"; & "ml\.venv-win\Scripts\python.exe" llm\scripts\gen_baseline_features.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "llm" / "src"))

from llm_ta import features as feat  # noqa: E402

GRID_CSV = REPO_ROOT / "data" / "features" / "decision_grid.csv"
OUT_CSV = REPO_ROOT / "data" / "news" / "baseline_features.csv"
WINDOW_HOURS = 7 * 24  # decision_grid news_window = trailing 7 calendar days

UNIVERSE = ["SBER", "GAZP", "LKOH", "GMKN", "ROSN", "NVTK",
            "TATN", "MGNT", "MTSS", "SNGS", "CHMF", "ALRS"]


def main() -> int:
    grid = pd.to_datetime(pd.read_csv(GRID_CSV)["as_of"])  # tz-aware MSK
    as_ofs = [feat.to_msk(ts.to_pydatetime()) for ts in grid]
    print(f"grid: {len(as_ofs)} as_of  ({as_ofs[0].date()} .. {as_ofs[-1].date()})")

    rows: list[dict] = []
    for ticker in UNIVERSE:
        disclosures = feat.load_disclosures(ticker)  # loaded once per ticker
        nonzero = 0
        for as_of in as_ofs:
            f, _ = feat.compute_features(disclosures, as_of, window_hours=WINDOW_HOURS)
            if f["news_count"] > 0:
                nonzero += 1
            rows.append({
                "ticker": ticker,
                "as_of": as_of.isoformat(),
                "sentiment": f["sentiment"],
                "impact_score": f["impact_score"],
                "novelty": f["novelty"],
                "event_type": f["event_type"],
                "news_count": f["news_count"],
                "recency_minutes": f["recency_minutes"],
            })
        print(f"  {ticker}: {len(as_ofs)} cells, {nonzero} with news "
              f"({nonzero / len(as_ofs):.0%})")

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"\nwrote {OUT_CSV}  rows={len(df)}")
    print(f"  cells with news: {(df['news_count'] > 0).mean():.1%}")
    print(f"  sentiment nonzero: {(df['sentiment'] != 0).mean():.1%}; "
          f"event_type=none: {(df['event_type'] == 'none').mean():.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
