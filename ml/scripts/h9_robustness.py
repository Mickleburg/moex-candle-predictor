"""H9 production P0 — hedge and parameter robustness.

Is the edge specific to a beta=1 IMOEX hedge and the exact entry/exit offsets, or robust? A
deployable sleeve must not hinge on one lucky choice. Compares hedge methods (IMOEX beta=1 /
beta-adjusted / sector-index / none) and sweeps entry/exit/vol-window/cap. Reports IS Sharpe (the
in-sample edge; forward stays thin -> shadow). Realistic costs from h9_cost_model applied throughout.
"""

from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

ML_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_DIR))

from scripts.xsec_eval_harness import load_daily_panel  # noqa: E402
from scripts.h9_dividend_research import load_daily as load_one  # noqa: E402
from scripts.h9_dividend_sleeve_sim import stats, UNIVERSE, FORWARD_START  # noqa: E402
from scripts.h9_cost_model import SLIP_TIER, COMMISSION, HEDGE_COST  # noqa: E402
from src.features.cross_sectional import _load_close  # noqa: E402  (1D-first loader, 1H fallback)
from src.service.dividend_sleeve import (  # noqa: E402
    load_dividend_calendar, active_window_map, inverse_vol_weights,
)

SECTOR = {"SBER": "MOEXFN", "VTBR": "MOEXFN", "GAZP": "MOEXOG", "LKOH": "MOEXOG", "ROSN": "MOEXOG",
          "NVTK": "MOEXOG", "TATN": "MOEXOG", "SNGS": "MOEXOG", "GMKN": "MOEXMM", "CHMF": "MOEXMM",
          "ALRS": "MOEXMM", "MAGN": "MOEXMM", "NLMK": "MOEXMM", "PLZL": "MOEXMM", "MGNT": "MOEXCN",
          "MTSS": "MOEXTL"}


def build_weights_p(panel, calendar, entry, exit_off, vw, mw) -> pd.DataFrame:
    idx = panel.index
    amap = active_window_map(idx, calendar, list(panel.columns), entry, exit_off)
    W = pd.DataFrame(0.0, index=idx, columns=panel.columns)
    for pos in range(vw, len(idx)):
        active = [t for t in panel.columns if pos in amap.get(t, set())]
        if active:
            for t, w in inverse_vol_weights(panel, active, pos, vw, mw).items():
                W.iat[pos, W.columns.get_loc(t)] = w
    return W


def costs(W) -> pd.Series:
    dW = (W - W.shift(1)).abs().fillna(0.0)
    rate = pd.Series({t: COMMISSION + SLIP_TIER.get(t, 6) / 1e4 for t in W.columns})
    return (dW * rate).sum(axis=1) + W.sum(axis=1).diff().abs().fillna(0.0) * HEDGE_COST


def pnl_hedged(panel, market, sectors, W, mode: str) -> pd.Series:
    rets = panel.pct_change(); held = W.shift(1).fillna(0.0)
    gross_long = (held * rets).sum(axis=1)
    mret = market.pct_change()
    if mode == "imoex1":
        hedge = held.sum(axis=1) * mret
    elif mode == "beta_adj":
        beta = panel.pct_change().rolling(60).cov(mret).div(mret.rolling(60).var(), axis=0).shift(1)
        hedge = (held * beta * mret.values[:, None]).sum(axis=1)
    elif mode == "sector":
        sret = sectors.pct_change()
        hedge = sum((held[t] * sret[SECTOR[t]]) for t in W.columns if SECTOR[t] in sectors.columns)
    elif mode == "none":
        hedge = 0.0
    return (gross_long - hedge - costs(W)).fillna(0.0)


def main() -> int:
    panel = load_daily_panel(UNIVERSE)
    market = load_one("IMOEX").reindex(panel.index).ffill()
    secs = sorted(set(SECTOR.values()))
    # sector indices: MOEXMM/CN/TL are 1D-only -> use the 1D-first loader (1H fallback)
    sectors = pd.DataFrame({s: _load_close(s, "1D") for s in secs}).reindex(panel.index).ffill()
    cal = load_dividend_calendar()

    print("H9 robustness — net of realistic costs (commission+tier slippage+hedge)\n")
    print("HEDGE method comparison (entry -12, exit -2, vol 20, cap 0.34):")
    W = build_weights_p(panel, cal, 12, 2, 20, 0.34)
    for mode in ("imoex1", "beta_adj", "sector", "none"):
        p = pnl_hedged(panel, market, sectors, W, mode)
        f = stats(p); i = stats(p, hi=FORWARD_START)
        print(f"  {mode:9} | full cum {f['cum']:+.3f} Sh {f['sharpe']:+.2f} DD {f['maxdd']:+.3f} "
              f"| IS Sh {i['sharpe']:+.2f}")

    print("\nPARAMETER sweep (IMOEX hedge), IS Sharpe / full Sharpe — looking for stability:")
    print(f"  {'entry':>5} {'exit':>4} {'vol':>4} {'cap':>4} | {'IS Sh':>6} {'full Sh':>7} {'full cum':>8}")
    for entry, exit_off, vw, mw in product((10, 12, 15), (2, 3), (20, 40), (0.34, 0.50)):
        Wp = build_weights_p(panel, cal, entry, exit_off, vw, mw)
        p = pnl_hedged(panel, market, sectors, Wp, "imoex1")
        f = stats(p); i = stats(p, hi=FORWARD_START)
        print(f"  {entry:>5} {exit_off:>4} {vw:>4} {mw:>4} | {i['sharpe']:>+6.2f} {f['sharpe']:>+7.2f} "
              f"{f['cum']:>+8.3f}")
    print("\nRead: edge should stay positive across hedge methods AND parameter choices (no single")
    print("lucky setting). Forward stays thin regardless -> shadow. is_production=false.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
