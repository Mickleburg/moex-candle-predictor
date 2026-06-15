"""H4 — Is realized volatility predictable? (sizing + regime-filter input)

Hypothesis (docs/RESEARCH_HYPOTHESES.md, H4): realized vol is far more predictable than
direction (vol clusters). If so it is useful for position SIZING (scale down when high vol
is expected) and as a REGIME filter (H5), even if direction stays unpredictable.

Method (daily, pooled over the 12-name universe, no lookahead):
    realized vol at t over window W:  rv[t]   = std(daily returns[t-W+1 .. t])
    target (future vol over horizon): fwd_rv  = std(daily returns[t+1 .. t+h])
    predictors (all PAST-ONLY):
        persistence : pred = rv[t]
        ewma        : exponentially-weighted vol (lambda=0.94, RiskMetrics)
        rollmean    : mean of rv over last 3 windows
    naive baseline : unconditional mean vol (no skill)
Metrics OUT-OF-SAMPLE: corr(pred, fwd_rv), R^2, QLIKE (lower better). Split IS (<2025) vs
FORWARD (>=2025). Success: positive, STABLE corr/R^2 on forward and QLIKE < naive ->
vol is usable for sizing/regime.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ML_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_DIR))

from scripts.xsec_eval_harness import UNIVERSE, FORWARD_START, load_daily_panel  # noqa: E402


def ewma_vol(returns: pd.Series, lam: float = 0.94) -> pd.Series:
    """RiskMetrics EWMA volatility (past-only)."""
    var = returns.pow(2).ewm(alpha=1 - lam, adjust=False).mean()
    return np.sqrt(var)


def build_samples(panel: pd.DataFrame, W: int, h: int) -> pd.DataFrame:
    """Pool (date, ticker) rows with past-only predictors and the future-vol target."""
    rows = []
    rets = panel.pct_change()
    for tkr in panel.columns:
        r = rets[tkr].dropna()
        rv = r.rolling(W).std()
        ew = ewma_vol(r)
        roll = rv.rolling(3).mean()
        fwd = r.shift(-1).rolling(h).std().shift(-(h - 1))  # std of r[t+1..t+h] placed at t
        df = pd.DataFrame({"t": r.index, "rv": rv.values, "ewma": ew.values,
                           "rollmean": roll.values, "fwd_rv": fwd.values})
        rows.append(df.dropna())
    return pd.concat(rows, ignore_index=True)


def qlike(actual_var: np.ndarray, pred_var: np.ndarray) -> float:
    pred_var = np.clip(pred_var, 1e-12, None)
    ratio = actual_var / pred_var
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def evaluate(samples: pd.DataFrame, predictor: str) -> dict:
    def metrics(df):
        a, p = df["fwd_rv"].to_numpy(), df[predictor].to_numpy()
        if len(df) < 10:
            return {}
        corr = float(np.corrcoef(a, p)[0, 1])
        ss_res = float(np.sum((a - p) ** 2))
        ss_tot = float(np.sum((a - a.mean()) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        ql = qlike(a ** 2, p ** 2)
        ql_naive = qlike(a ** 2, np.full_like(a, a.mean()) ** 2)
        return {"corr": round(corr, 3), "r2": round(r2, 3),
                "qlike": round(ql, 4), "qlike_naive": round(ql_naive, 4), "n": len(df)}

    is_df = samples[samples["t"] < FORWARD_START]
    fw_df = samples[samples["t"] >= FORWARD_START]
    return {"predictor": predictor, "is": metrics(is_df), "fwd": metrics(fw_df)}


def main() -> int:
    panel = load_daily_panel()
    print(f"Daily panel: {panel.shape[1]} tickers x {len(panel)} days, "
          f"forward>={FORWARD_START.date()}\n")
    print("=== H4: realized-vol predictability (pooled, out-of-sample) ===")
    for W in (10, 20):
        for h in (5, 10, 20):
            samples = build_samples(panel, W=W, h=h)
            print(f"\n-- window W={W}, horizon h={h}  (n={len(samples)}) --")
            print(f"{'predictor':10} | {'IS corr':>7} {'IS r2':>6} {'IS QLIKE':>8} | "
                  f"{'FWD corr':>8} {'FWD r2':>6} {'FWD QLIKE':>9} {'naive':>7}")
            for pred in ("persistence_rv", "ewma", "rollmean"):
                col = "rv" if pred == "persistence_rv" else pred
                m = evaluate(samples.rename(columns={"rv": "rv"}), col)
                i, f = m["is"], m["fwd"]
                print(f"{pred:10} | {i.get('corr',0):>7} {i.get('r2',0):>6} {i.get('qlike',0):>8} | "
                      f"{f.get('corr',0):>8} {f.get('r2',0):>6} {f.get('qlike',0):>9} "
                      f"{f.get('qlike_naive',0):>7}")
    print("\nReading: high, stable FWD corr and QLIKE < naive => vol is predictable and usable")
    print("for position sizing (scale inversely to predicted vol) and as a regime filter (H5).")
    print("This is INFORMATION for risk_manager sizing, not a directional edge.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
