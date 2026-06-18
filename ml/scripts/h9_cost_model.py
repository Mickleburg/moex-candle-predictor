"""H9 production P0 — realistic transaction-cost model + net-of-cost re-validation.

The sleeve backtest charged only 5bps commission. Production must survive the FULL cost stack:
commission + SLIPPAGE (bid/ask + impact, larger for less liquid names) + the SHORT-HEDGE cost
(IMOEX futures) + tax. This script re-prices the same book under a parametrized cost model and runs
a BREAKEVEN sweep: at what total round-trip cost does the edge die, and how far is that from a
realistic estimate? An edge that only survives at unrealistically low cost is not deployable.

Approach: per day, stock-leg turnover and hedge turnover are charged separately. Commission is known
(5bps/side); slippage is swept (and a per-liquidity-tier realistic case is shown); the IMOEX-futures
hedge is cheap (~2-3bps/side, very liquid) with carry treated as ~neutral (in a high-rate regime the
short future even earns positive carry — we conservatively ignore that benefit). Tax (13% on net
positive P&L) is applied as a final haircut. is_production=false.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ML_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_DIR))

from scripts.xsec_eval_harness import load_daily_panel  # noqa: E402
from scripts.h9_dividend_research import load_daily as load_one  # noqa: E402
from scripts.h9_dividend_sleeve_sim import build_weights, stats, UNIVERSE, FORWARD_START  # noqa: E402
from src.service.dividend_sleeve import load_dividend_calendar  # noqa: E402

COMMISSION = 0.0005          # broker commission per side (known)
HEDGE_COST = 0.0003          # IMOEX-futures per side (commission+slippage; very liquid)
TAX_RATE = 0.13              # RU capital-gains tax on net positive P&L (final haircut)

# slippage per side by liquidity tier (bps). Blue chips tight; less liquid names wider.
SLIP_TIER = {
    "SBER": 3, "GAZP": 3, "LKOH": 3, "GMKN": 4, "ROSN": 4, "NVTK": 5, "TATN": 5, "MGNT": 6,
    "CHMF": 6, "MTSS": 7, "SNGS": 7, "ALRS": 8, "NLMK": 8, "MAGN": 9, "PLZL": 6, "VTBR": 8,
}


def book_pnl(panel, market, W, slip_oneway_by_name: dict | float, commission=COMMISSION,
             hedge_cost=HEDGE_COST) -> pd.Series:
    """Daily net P&L of the hedged book under the cost model (per-name slippage)."""
    rets = panel.pct_change(); mret = market.pct_change()
    held = W.shift(1).fillna(0.0)
    gross = (held * rets).sum(axis=1) - held.sum(axis=1) * mret      # pre-cost, beta=1 hedge
    dW = (W - W.shift(1)).abs().fillna(0.0)
    if isinstance(slip_oneway_by_name, dict):
        stock_cost_rate = pd.Series({t: commission + slip_oneway_by_name.get(t, 6) / 1e4
                                     for t in W.columns})
        stock_cost = (dW * stock_cost_rate).sum(axis=1)
    else:
        stock_cost = dW.sum(axis=1) * (commission + slip_oneway_by_name)
    hedge_turnover = W.sum(axis=1).diff().abs().fillna(0.0)
    hedge_c = hedge_turnover * hedge_cost
    return (gross - stock_cost - hedge_c).fillna(0.0)


def after_tax(pnl: pd.Series) -> pd.Series:
    """Tax on NET ANNUAL gains (losses offset within the year), not per-day. Haircut each positive
    year's P&L by TAX_RATE; losing years untaxed (no cross-year carry-forward credit, conservative)."""
    out = pnl.copy()
    for _, idx in pnl.groupby(pnl.index.year).groups.items():
        seg = pnl.loc[idx]
        if seg.sum() > 0:
            out.loc[idx] = seg * (1 - TAX_RATE)
    return out


def line(label, pnl):
    f = stats(pnl); i = stats(pnl, hi=FORWARD_START)
    print(f"  {label:34} | full cum {f['cum']:+.3f} Sh {f['sharpe']:+.2f} DD {f['maxdd']:+.3f} "
          f"| IS Sh {i['sharpe']:+.2f}")


def main() -> int:
    panel = load_daily_panel(UNIVERSE)
    market = load_one("IMOEX").reindex(panel.index).ffill()
    W = build_weights(panel, load_dividend_calendar())
    print("H9 cost model — net-of-cost re-validation (hedged dividend run-up book)\n")

    print("Flat-slippage BREAKEVEN sweep (commission 5bps + slippage/side, hedge 3bps/side):")
    for slip in (0, 3, 5, 8, 12, 16, 20, 25):
        line(f"slippage {slip:>2}bps/side (RT~{2*(5+slip)+2*3}bps)", book_pnl(panel, market, W, slip / 1e4))

    print("\nRealistic per-name tier (SBER/GAZP/LKOH 3 .. MAGN 9 bps/side):")
    real = book_pnl(panel, market, W, SLIP_TIER)
    line("realistic, pre-tax", real)
    line("realistic, AFTER 13% tax", after_tax(real))

    # locate breakeven flat slippage where IS cum crosses 0
    grid = np.arange(0, 0.0040, 0.0002)
    is_cum = [stats(book_pnl(panel, market, W, s), hi=FORWARD_START)["cum"] for s in grid]
    be = next((grid[k] for k in range(len(grid)) if is_cum[k] <= 0), None)
    print("\nBreakeven flat slippage (IS cum -> 0): "
          f"{'%.0f bps/side' % (be*1e4) if be is not None else '>40 bps/side (never within grid)'}")
    print("Realistic slippage is ~3-9 bps/side -> compare to breakeven for the margin of safety.")
    print("Read: edge is deployable only if realistic cost sits well below breakeven. is_production=false.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
