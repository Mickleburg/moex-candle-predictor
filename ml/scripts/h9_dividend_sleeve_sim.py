"""H9 deployable sleeve — PORTFOLIO backtest of the dividend run-up (sleeve S3-adjacent).

Stage 2 measured per-EVENT returns. This runs the actual deployable BOOK: each day hold the basket
of names currently in their pre-ex window (offsets [-12,-2]), inverse-vol sized (H4) and capped,
market-hedged (beta=1 short IMOEX), with fees on turnover. Reports the equity curve full / in-sample
/ forward / per-year, drawdown, and hedged vs unhedged (to show the hedge earns its keep). Past-only;
uses the serving logic in `src/service/dividend_sleeve.py` so the backtest and live path agree.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ML_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_DIR))

from scripts.xsec_eval_harness import load_daily_panel  # noqa: E402
from scripts.h9_dividend_research import load_daily as load_daily_one  # noqa: E402
from src.service.dividend_sleeve import (  # noqa: E402
    load_dividend_calendar, active_window_map, inverse_vol_weights,
    ENTRY_OFFSET, EXIT_OFFSET, VOL_WINDOW,
)
from src.service.dividend_universe import active_universe, FORWARD_START  # noqa: E402

# Single-source universe; env H9_UNIVERSE=expanded flips the whole research toolchain at once.
UNIVERSE = active_universe()
FEE_ONEWAY = 0.0005
BARS_PER_YEAR = 247


def build_weights(panel: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """Daily long-weight matrix [days x tickers] from the sleeve serving logic (past-only)."""
    idx = panel.index
    amap = active_window_map(idx, calendar, list(panel.columns), ENTRY_OFFSET, EXIT_OFFSET)
    W = pd.DataFrame(0.0, index=idx, columns=panel.columns)
    for pos in range(VOL_WINDOW, len(idx)):
        active = [t for t in panel.columns if pos in amap.get(t, set())]
        if not active:
            continue
        for t, w in inverse_vol_weights(panel, active, pos).items():
            W.iat[pos, W.columns.get_loc(t)] = w
    return W


def stats(s: pd.Series, lo=None, hi=None) -> dict:
    if lo is not None: s = s[s.index >= lo]
    if hi is not None: s = s[s.index < hi]
    s = s.dropna()
    if len(s) == 0 or s.std() == 0:
        return {"cum": float(s.sum()) if len(s) else 0.0, "sharpe": 0.0, "maxdd": 0.0}
    eq = (1 + s).cumprod()
    return {"cum": float(eq.iloc[-1] - 1),
            "sharpe": float(s.mean() / s.std() * np.sqrt(BARS_PER_YEAR)),
            "maxdd": float((eq / eq.cummax() - 1).min())}


def main() -> int:
    panel = load_daily_panel(UNIVERSE)
    market = load_daily_one("IMOEX").reindex(panel.index).ffill()
    calendar = load_dividend_calendar()
    W = build_weights(panel, calendar)

    rets = panel.pct_change()
    mret = market.pct_change()
    held = W.shift(1).fillna(0.0)
    gross_long = (held * rets).sum(axis=1)                 # long-leg P&L
    net_hedged = gross_long - held.sum(axis=1) * mret       # minus beta=1 IMOEX hedge
    turnover = (W - W.shift(1)).abs().sum(axis=1).fillna(0.0)
    fee = turnover * FEE_ONEWAY * 2.0                       # stock leg + its hedge share
    pnl_hedged = (net_hedged - fee).fillna(0.0)
    pnl_unhedged = (gross_long - fee).fillna(0.0)

    invested = (held.sum(axis=1) > 0)
    print(f"H9 dividend run-up SLEEVE — portfolio backtest "
          f"({panel.index.min().date()}..{panel.index.max().date()}, fee {FEE_ONEWAY*1e4:.0f}bps)")
    print(f"  entry -{ENTRY_OFFSET}/exit -{EXIT_OFFSET}, inv-vol sized, beta=1 IMOEX hedge")
    print(f"  days invested {invested.mean():.0%}; avg active names when invested "
          f"{held[invested].gt(0).sum(axis=1).mean():.1f}\n")

    for label, pnl in [("market-HEDGED", pnl_hedged), ("unhedged (long-only)", pnl_unhedged)]:
        f = stats(pnl); i = stats(pnl, hi=FORWARD_START); w = stats(pnl, lo=FORWARD_START)
        print(f"  {label:22} | full cum {f['cum']:+.3f} Sh {f['sharpe']:+.2f} DD {f['maxdd']:+.3f} "
              f"| IS Sh {i['sharpe']:+.2f} | FWD cum {w['cum']:+.3f} Sh {w['sharpe']:+.2f}")

    print("\n  per-year (market-hedged) Sharpe / cum:")
    for y, s in pnl_hedged.groupby(pnl_hedged.index.year):
        st = stats(s)
        print(f"    {y}: Sh {st['sharpe']:+.2f}  cum {st['cum']:+.3f}")
    print("\nRead: hedged book should beat unhedged on Sharpe/DD (hedge removes market beta from the")
    print("hold). Forward is thin (few 2025 events) — accrue via shadow. is_production=false.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
