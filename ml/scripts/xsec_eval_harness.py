"""Cross-sectional evaluation harness (H2/H3) — the rig that judges whether a feature
(price OR news) carries market-neutral cross-sectional alpha at a daily horizon.

WHY THIS EXISTS
    H1a proved price-only cross-section has no economic edge (see xsec_h1_baseline report).
    The V2 thesis: alpha must come from EXOGENOUS info (news). To test that honestly we need
    one fixed, reusable rig that scores ANY feature/model the same way, so "news beats price"
    is a like-for-like claim. This module IS that rig. The news chat builds features TO this
    interface; the benchmark below is the number news must beat.

WHAT IT MEASURES
    * Rank IC (Spearman) — per decision date, correlation between a ticker's cross-sectional
      SCORE and its realized RELATIVE forward return (return minus universe mean). This is the
      standard cross-sectional alpha diagnostic: positive, stable IC => the score ranks names.
      Cheap to compute for a candidate news feature BEFORE any full backtest.
    * Market-neutral backtest — long top-k by score / short bottom-k, non-overlapping rebalance
      every H days, fees applied; reported IN-SAMPLE (<2025) vs FORWARD (>=2025).

INTERFACE FOR THE NEWS CHAT
    A "score function" maps (panel, t_index) -> np.ndarray of one score per universe ticker,
    using ONLY information available at decision time t (no lookahead). Price example below.
    News features arrive as llm_analysis/feature_bundle -> wrap them as a score_fn (e.g.
    sentiment*impact, cross-sectionally ranked) and call evaluate_scores(). Same rig, same
    metrics, directly comparable to the price benchmark.

NO-LOOKAHEAD
    Features at t use panel.iloc[..t]. The label uses close[t+H] (future) only as the outcome.
    For news: sources[].published_at <= decision time (invariant #3) — enforced upstream.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

ML_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = ML_DIR.parent
sys.path.insert(0, str(ML_DIR))

from src.data.load import to_moscow_time  # noqa: E402

DATA_RAW = REPO_ROOT / "data" / "raw"
UNIVERSE = ["SBER", "GAZP", "LKOH", "GMKN", "ROSN", "NVTK",
            "TATN", "MGNT", "MTSS", "SNGS", "CHMF", "ALRS"]
FEE_ONEWAY = 0.0005
FORWARD_START = pd.Timestamp("2025-01-01", tz="Europe/Moscow")

# A score function: (daily_close_panel, t_index) -> array[n_tickers], PAST-ONLY.
ScoreFn = Callable[[pd.DataFrame, int], np.ndarray]


def load_daily_panel(universe: list[str] = UNIVERSE) -> pd.DataFrame:
    """Daily last-close matrix (time x ticker) on the timestamp intersection."""
    series = {}
    for tkr in universe:
        files = sorted(DATA_RAW.glob(f"{tkr}_1H_*.parquet"))
        if not files:
            print(f"  WARN: no parquet for {tkr}")
            continue
        df = pd.read_parquet(files[-1])
        df.columns = [c.lower() for c in df.columns]
        s = pd.Series(df["close"].to_numpy(float), index=to_moscow_time(df["begin"]))
        s = s[~s.index.duplicated(keep="last")].sort_index()
        series[tkr] = s
    mat = pd.DataFrame(series).resample("1D").last().dropna(how="any")
    return mat


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return 0.0
    ra = pd.Series(a).rank().to_numpy(dtype=float)
    rb = pd.Series(b).rank().to_numpy(dtype=float)
    ra = ra - ra.mean(); rb = rb - rb.mean()
    denom = np.sqrt((ra @ ra) * (rb @ rb))
    return float(ra @ rb / denom) if denom > 0 else 0.0


def evaluate_scores(panel: pd.DataFrame, score_fn: ScoreFn, horizon: int,
                    k: int = 3, fee_oneway: float = FEE_ONEWAY,
                    label: str = "score") -> dict:
    """Run rank-IC + market-neutral backtest for a score function. Returns a metrics dict."""
    n = len(panel)
    fee = 4.0 * fee_oneway                       # gross=2, open+close round-trip per period
    ics, rows = [], []
    t = 0
    # IC uses every valid date (overlapping ok for a diagnostic); backtest steps by horizon.
    next_trade = None
    for t in range(n - horizon):
        scores = score_fn(panel, t)
        if scores is None or not np.all(np.isfinite(scores)):
            continue
        fwd = panel.iloc[t + horizon].to_numpy() / panel.iloc[t].to_numpy() - 1.0
        fwd_rel = fwd - fwd.mean()               # market-neutral (relative) outcome
        ics.append((panel.index[t], _spearman(scores, fwd_rel)))
        if next_trade is None or t >= next_trade:
            order = np.argsort(scores)
            ret = fwd[order[-k:]].mean() - fwd[order[:k]].mean() - fee
            rows.append((panel.index[t], ret))
            next_trade = t + horizon
    ic_df = pd.DataFrame(ics, columns=["t", "ic"]).set_index("t")
    bt = pd.DataFrame(rows, columns=["t", "ret"]).set_index("t")

    def split(df, col):
        return df[df.index < FORWARD_START][col], df[df.index >= FORWARD_START][col]

    ic_is, ic_fw = split(ic_df, "ic")
    bt_is, bt_fw = split(bt, "ret")

    def cum(s): return float((1 + s).prod() - 1) if len(s) else 0.0
    def icr(s): return (round(float(s.mean()), 4), round(float(s.mean() / (s.std() + 1e-9)), 3))

    ic_is_m, ic_is_ir = icr(ic_is)
    ic_fw_m, ic_fw_ir = icr(ic_fw)
    return {
        "label": label, "horizon": horizon, "k": k,
        "ic_all": round(float(ic_df["ic"].mean()), 4),
        "ic_is_mean": ic_is_m, "ic_is_ir": ic_is_ir,
        "ic_fwd_mean": ic_fw_m, "ic_fwd_ir": ic_fw_ir,
        "bt_is_cum": round(cum(bt_is), 4), "bt_fwd_cum": round(cum(bt_fw), 4),
        "bt_fwd_win": round(float((bt_fw > 0).mean()), 3) if len(bt_fw) else 0.0,
        "n_ic": int(len(ic_df)), "n_trades_fwd": int(len(bt_fw)),
    }


# ---- price-only baseline score functions (the benchmark news must beat) ----

def momentum_score(lookback: int) -> ScoreFn:
    def fn(panel: pd.DataFrame, t: int) -> np.ndarray | None:
        if t < lookback:
            return None
        return panel.iloc[t].to_numpy() / panel.iloc[t - lookback].to_numpy() - 1.0
    return fn


def print_row(m: dict) -> None:
    print(f"{m['label']:22} H={m['horizon']:>2} k={m['k']} | "
          f"IC all={m['ic_all']:+.4f}  IS={m['ic_is_mean']:+.4f}(IR{m['ic_is_ir']:+.2f})  "
          f"FWD={m['ic_fwd_mean']:+.4f}(IR{m['ic_fwd_ir']:+.2f}) | "
          f"bt IS={m['bt_is_cum']:+.3f} FWD={m['bt_fwd_cum']:+.3f} win={m['bt_fwd_win']:.2f} "
          f"(n_ic={m['n_ic']}, n_fwd={m['n_trades_fwd']})")


def main() -> int:
    print("Loading daily panel...")
    panel = load_daily_panel()
    print(f"  {panel.shape[1]} tickers x {len(panel)} days, "
          f"{panel.index.min().date()}..{panel.index.max().date()}")
    print(f"  forward from {FORWARD_START.date()}\n")
    print("=== PRICE-ONLY BENCHMARK (momentum) — the number news features must beat ===")
    for L in (5, 10, 20, 60):
        for H in (5, 10, 20):
            print_row(evaluate_scores(panel, momentum_score(L), horizon=H, k=3,
                                      label=f"mom_L{L}"))
    print("\nReading: a feature has cross-sectional alpha when FWD IC mean is positive AND")
    print("stable (IR>0) AND the forward backtest is positive net of fees. Price momentum")
    print("is the baseline; the news score_fn must beat its FORWARD IC to justify fusion.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
