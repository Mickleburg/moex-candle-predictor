"""News-feature -> score_fn adapter + harness self-test (step 2 of H2 prep).

Purpose
    Make plugging real news features into the cross-sectional rig INSTANT and de-risk the
    integration before the LLM chat delivers data. Real llm_analysis outputs arrive as a tidy
    per-(ticker, as_of) table; this adapter turns that table into a `score_fn(panel, t)` the
    eval harness already understands, with no-lookahead enforced by `as_of <= decision time`.

    Also a SELF-TEST of the harness: feed a synthetic feature with KNOWN correlation to the
    forward relative return and confirm the rig recovers a positive rank IC (~the injected
    strength); feed pure noise and confirm IC ~ 0. This proves the rig measures signal and
    does not manufacture it — so when real news scores positive IC, we trust it.

News table schema (matches per-ticker llm_analysis.features rows):
    columns: ticker, as_of (tz-aware datetime), + feature columns (sentiment, impact_score, ...)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ML_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_DIR))

from scripts.xsec_eval_harness import (  # noqa: E402
    UNIVERSE, FORWARD_START, load_daily_panel, evaluate_scores,
)


def build_news_score_fn(news: pd.DataFrame, score_col: str,
                        max_staleness_days: int = 5):
    """Return score_fn(panel, t): per-ticker latest `score_col` with as_of <= decision date.

    No-lookahead: only news with as_of <= panel.index[t] is visible. News older than
    `max_staleness_days` is dropped (stale -> neutral 0). Tickers with no fresh news -> 0.
    """
    news = news.sort_values("as_of")
    by_ticker = {tkr: g for tkr, g in news.groupby("ticker")}

    def score_fn(panel: pd.DataFrame, t: int):
        decision = panel.index[t]
        floor = decision - pd.Timedelta(days=max_staleness_days)
        out = np.zeros(panel.shape[1], dtype=float)
        for j, tkr in enumerate(panel.columns):
            g = by_ticker.get(tkr)
            if g is None:
                continue
            visible = g[(g["as_of"] <= decision) & (g["as_of"] > floor)]
            if len(visible):
                out[j] = float(visible.iloc[-1][score_col])
        return out

    return score_fn


# ---------- synthetic generators (rig self-test only; NOT a real result) ----------

def _forward_rel(panel: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Per (date, ticker) realized forward RELATIVE return — the target the rig scores against."""
    fwd = panel.shift(-horizon) / panel - 1.0
    return fwd.sub(fwd.mean(axis=1), axis=0)


def synthetic_news(panel: pd.DataFrame, horizon: int, signal: float,
                   coverage: float = 0.6, seed: int = 7) -> pd.DataFrame:
    """Build a synthetic news table whose feature = signal*z(forward_rel) + noise.

    `signal` in [0,1] sets how much of the feature is genuine forward info (deliberate
    lookahead — this is a controlled rig test). `coverage` = fraction of (date,ticker) cells
    that get a news item (sparse, like reality). as_of is set 1h before the decision date so
    the item is visible at t (no same-day leakage in the daily rig).
    """
    rng = np.random.default_rng(seed)
    rel = _forward_rel(panel, horizon)
    relz = rel.sub(rel.mean(axis=1), axis=0).div(rel.std(axis=1).replace(0, np.nan), axis=0)
    rows = []
    for i, date in enumerate(panel.index):
        for j, tkr in enumerate(panel.columns):
            if rng.random() > coverage:
                continue
            base = relz.iat[i, j]
            if not np.isfinite(base):
                base = 0.0
            feat = signal * base + (1.0 - signal) * rng.standard_normal()
            rows.append({"ticker": tkr, "as_of": date - pd.Timedelta(hours=1),
                         "sentiment": float(np.tanh(feat)), "impact_score": float(rng.random())})
    return pd.DataFrame(rows)


def main() -> int:
    panel = load_daily_panel()
    H = 20
    print(f"Panel: {panel.shape[1]} tickers x {len(panel)} days; horizon={H}, "
          f"forward>={FORWARD_START.date()}\n")
    print("=== HARNESS SELF-TEST (synthetic news -> adapter -> rig) ===")
    print("Expect: IC rises with injected signal; pure noise ~ 0. Validates the integration.\n")
    for sig in (0.0, 0.05, 0.15, 0.30):
        news = synthetic_news(panel, horizon=H, signal=sig, seed=7)
        fn = build_news_score_fn(news, score_col="sentiment", max_staleness_days=5)
        m = evaluate_scores(panel, fn, horizon=H, k=3, label=f"synthetic_sig={sig:.2f}")
        print(f"  signal={sig:.2f} | news_rows={len(news):5d} | "
              f"IC all={m['ic_all']:+.4f}  IS={m['ic_is_mean']:+.4f}  FWD={m['ic_fwd_mean']:+.4f}"
              f"(IR{m['ic_fwd_ir']:+.2f}) | bt FWD={m['bt_fwd_cum']:+.3f}")
    print("\nInterpretation: signal=0 must give IC~0 (rig invents nothing); higher injected")
    print("signal must lift FWD IC monotonically. If so, the rig + adapter are correct and")
    print("REAL llm_analysis features can be plugged via build_news_score_fn() unchanged.")
    print("Price benchmark to beat (xsec_eval_harness): FWD IC ~0.05. ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
