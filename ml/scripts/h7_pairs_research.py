"""H7 — Sector pairs statistical arbitrage (sleeve S1) — foundational screen.

WHY THIS IS DIFFERENT FROM H1/H2/H6
    Those all tried cross-sectional factor RANKING of 12 names (price, news, macro) and all failed
    the same way (few names, dominant beta, flat 2025 forward). H7 changes the MECHANISM: pairwise
    mean-reversion of a cointegrated spread. Beta is removed CONSTRUCTIVELY (long one leg, short
    beta*the other), not by ranking. This is the last structurally-different market-neutral idea; if
    it also fails, market-neutral alpha is not available on this universe.

THE HONEST GATE (no-lookahead)
    1. Hedge ratio beta from OLS log(a)~log(b) on IN-SAMPLE only (<2025).
    2. Spread = log(a) - beta*log(b). Mean-reversion screened in-sample (AR(1) half-life).
    3. Pair SELECTION uses in-sample only. The forward period (>=2025) is the untouched test.
    4. Trade a past-only rolling z-score of the spread (entry |z|>2, exit |z|<0.5); fees on turnover.
    5. A pair counts only if, selected in-sample, it KEEPS reverting (positive net) out-of-sample.
       Aggregate = equal-weight book of selected pairs; judge by FORWARD net P&L, market-neutral.

    Statsmodels is unavailable on this env; Engle-Granger is hand-rolled (OLS + AR(1) half-life).
    The decisive test is out-of-sample reversion, not an ADF p-value.
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ML_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_DIR))

from scripts.xsec_eval_harness import load_daily_panel  # noqa: E402

FORWARD_START = pd.Timestamp("2025-01-01", tz="Europe/Moscow")
# Calibrate/select on the POST-shock stationary regime (2023-2024). The 2022 structural break is a
# different regime (H5 detects it cleanly) that a real pairs book gates OFF — excluding it is decided
# a priori by the regime, NOT by peeking at the 2025+ forward (which stays the untouched test).
IS_SELECT_START = pd.Timestamp("2023-01-01", tz="Europe/Moscow")
FEE_ONEWAY = 0.0005
BARS_PER_YEAR = 247

# within-sector clusters that actually have >=2 liquid names (pairs need same-sector partners)
CLUSTERS = {
    "energy": ["LKOH", "ROSN", "TATN", "NVTK", "SNGS", "GAZP"],
    "metals": ["GMKN", "CHMF", "ALRS"],
}
Z_WINDOW = 60
Z_ENTRY, Z_EXIT = 2.0, 0.5


def ols_hedge(a: pd.Series, b: pd.Series) -> float:
    """beta in log(a) = alpha + beta*log(b), least squares (in-sample slice passed in)."""
    x = np.log(b.to_numpy()); y = np.log(a.to_numpy())
    x1 = np.vstack([np.ones_like(x), x]).T
    beta = np.linalg.lstsq(x1, y, rcond=None)[0][1]
    return float(beta)


def half_life(spread: pd.Series) -> float:
    """AR(1) mean-reversion half-life (days). <0 or huge => not mean-reverting."""
    s = spread.dropna()
    ds = s.diff().dropna()
    lag = s.shift(1).dropna().loc[ds.index]
    x1 = np.vstack([np.ones(len(lag)), lag.to_numpy()]).T
    coef = np.linalg.lstsq(x1, ds.to_numpy(), rcond=None)[0][1]
    return float(-np.log(2) / coef) if coef < 0 else np.inf


def backtest_spread(spread: pd.Series, fee: float = FEE_ONEWAY) -> pd.Series:
    """Past-only rolling-z reversion trade on a log-spread. Daily P&L series (dollar-neutral pair)."""
    mu = spread.rolling(Z_WINDOW).mean()
    sd = spread.rolling(Z_WINDOW).std()
    z = (spread - mu) / sd
    dspread = spread.diff()                      # ~ ret_a - beta*ret_b
    pos = pd.Series(0.0, index=spread.index)
    cur = 0.0
    pvals = []
    for i in range(len(spread)):
        zi = z.iloc[i]
        if np.isfinite(zi):
            if cur == 0.0 and abs(zi) > Z_ENTRY:
                cur = -np.sign(zi)               # fade the deviation
            elif cur != 0.0 and abs(zi) < Z_EXIT:
                cur = 0.0
        pos.iloc[i] = cur
    # P&L at t uses position held at t-1 applied to dspread at t; fee on position change
    held = pos.shift(1).fillna(0.0)
    turnover = pos.diff().abs().fillna(0.0)
    pnl = held * dspread.fillna(0.0) - turnover * fee * 2.0   # 2 legs traded
    return pnl


def stats(pnl: pd.Series, lo=None, hi=None) -> dict:
    s = pnl
    if lo is not None: s = s[s.index >= lo]
    if hi is not None: s = s[s.index < hi]
    s = s.dropna()
    if len(s) == 0 or s.std() == 0:
        return {"cum": 0.0, "sharpe": 0.0, "n": len(s)}
    return {"cum": float(s.sum()), "sharpe": float(s.mean() / s.std() * np.sqrt(BARS_PER_YEAR)),
            "n": int((s != 0).sum())}


def main() -> int:
    panel = load_daily_panel()
    is_mask = (panel.index >= IS_SELECT_START) & (panel.index < FORWARD_START)
    print(f"H7 pairs stat-arb screen - {panel.shape[1]} names, "
          f"{panel.index.min().date()}..{panel.index.max().date()}, forward from {FORWARD_START.date()}")
    print(f"  hedge ratio + selection on POST-SHOCK in-sample {IS_SELECT_START.date()}..{FORWARD_START.date()}; "
          f"forward = untouched test\n")
    print(f"{'pair':14} {'beta':>6} {'half-life':>9} | {'IS cum':>7} {'IS Sh':>6} | "
          f"{'FWD cum':>7} {'FWD Sh':>6} {'sel':>4}")

    selected = []
    all_pnl = {}
    for sector, names in CLUSTERS.items():
        names = [n for n in names if n in panel.columns]
        for a, b in combinations(names, 2):
            sa, sb = panel[a], panel[b]
            beta = ols_hedge(sa[is_mask], sb[is_mask])
            if beta <= 0:                        # negative hedge = not a sane long/short pair
                continue
            spread = np.log(sa) - beta * np.log(sb)
            hl = half_life(spread[is_mask])
            pnl = backtest_spread(spread)
            all_pnl[f"{a}/{b}"] = pnl
            iss = stats(pnl, lo=IS_SELECT_START, hi=FORWARD_START)
            fws = stats(pnl, lo=FORWARD_START)
            # selection: mean-reverts in-sample (sane half-life) AND positive in-sample net
            sel = (2 <= hl <= 60) and iss["sharpe"] > 0.3
            if sel:
                selected.append(f"{a}/{b}")
            print(f"{a+'/'+b:14} {beta:6.2f} {hl:9.1f} | {iss['cum']:+7.3f} {iss['sharpe']:6.2f} | "
                  f"{fws['cum']:+7.3f} {fws['sharpe']:6.2f} {'YES' if sel else '':>4}")

    print(f"\nSelected in-sample (half-life 2..60d, IS Sharpe>0.3): {len(selected)} pairs: {selected}")
    if selected:
        book = pd.concat([all_pnl[p] for p in selected], axis=1).mean(axis=1)  # equal-weight book
        bi, bf = stats(book, lo=IS_SELECT_START, hi=FORWARD_START), stats(book, lo=FORWARD_START)
        print(f"\nEQUAL-WEIGHT BOOK of selected pairs:")
        print(f"  IN-SAMPLE  cum {bi['cum']:+.3f}  Sharpe {bi['sharpe']:+.2f}")
        print(f"  FORWARD    cum {bf['cum']:+.3f}  Sharpe {bf['sharpe']:+.2f}   <-- the honest gate")
        print(f"\nRead: forward Sharpe>0 on pairs SELECTED in-sample = the spread keeps reverting")
        print(f"out-of-sample = a real market-neutral edge. Forward<=0 = in-sample selection is overfit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
