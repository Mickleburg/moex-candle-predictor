"""H9 production (optional) — capacity analysis: how much AUM the sleeve absorbs.

The P0 cost model assumed slippage of 3-9 bps/side. That holds only while our orders are small vs
each name's liquidity. This estimates, per name, the position size at which market impact reaches
that assumed slippage, and backs out the book AUM where the P0 edge stays valid. Beyond it, impact
grows toward the breakeven (~34 bps/side) and the edge degrades.

Model (transparent, order-of-magnitude): ADV = median daily RUB turnover (last ~252 sessions).
Square-root impact: impact_bps = K * sqrt(participation), participation = daily_order / ADV, K=38
(=> ~12 bps at 10% ADV, ~4 bps at 1%). Each entry/exit is worked over SPREAD_DAYS sessions to cut
impact. A name at book weight w with AUM A trades A*w over the window -> daily_order = A*w/SPREAD_DAYS.
Capacity per name (at its assumed slippage tier) -> book AUM via the cap weight. Assumptions stated;
treat as a ballpark, not a guarantee.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ML_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = ML_DIR.parent
sys.path.insert(0, str(ML_DIR))

from scripts.h9_cost_model import SLIP_TIER  # noqa: E402
from src.service.dividend_sleeve import MAX_WEIGHT  # noqa: E402

DATA_RAW = REPO_ROOT / "data" / "raw"
UNIVERSE = list(SLIP_TIER.keys())
K_IMPACT = 38.0          # impact_bps = K*sqrt(participation_fraction)
SPREAD_DAYS = 3          # work each entry/exit over this many sessions
ADV_WINDOW = 252


def adv_rub(ticker: str) -> float | None:
    files = sorted(DATA_RAW.glob(f"{ticker}_1H_*.parquet"))
    if not files:
        return None
    df = pd.read_parquet(files[-1]); df.columns = [c.lower() for c in df.columns]
    if "value" not in df.columns:
        return None
    s = pd.Series(df["value"].to_numpy(float), index=pd.to_datetime(df["begin"]))
    daily = s.resample("1D").sum()
    daily = daily[daily > 0]
    return float(daily.tail(ADV_WINDOW).median())


def main() -> int:
    print("H9 sleeve capacity (assumptions: sqrt impact K=38, execute over "
          f"{SPREAD_DAYS} sessions, cap weight {MAX_WEIGHT})\n")
    print(f"{'name':5} {'ADV (RUB/day)':>16} {'slip tier':>10} {'max daily order':>16} "
          f"{'max position':>14} {'book AUM cap':>14}")
    rows = []
    for t in UNIVERSE:
        adv = adv_rub(t)
        if adv is None:
            continue
        slip = SLIP_TIER[t]                                   # bps/side we assumed in P0
        part_max = (slip / K_IMPACT) ** 2                     # participation giving that impact
        max_daily = part_max * adv
        max_pos = max_daily * SPREAD_DAYS                     # position worked over SPREAD_DAYS
        aum_cap = max_pos / MAX_WEIGHT                         # if this name takes the cap weight
        rows.append((t, adv, slip, max_daily, max_pos, aum_cap))
        print(f"{t:5} {adv:16,.0f} {slip:>9}b {max_daily:16,.0f} {max_pos:14,.0f} {aum_cap:14,.0f}")

    aums = sorted(r[5] for r in rows)
    binding = rows[int(np.argmin([r[5] for r in rows]))]
    print(f"\nBinding (least capacity) name: {binding[0]} -> book AUM ~ {binding[5]/1e6:,.0f} M RUB")
    print(f"Median per-name book-AUM capacity: ~{np.median(aums)/1e6:,.0f} M RUB")
    print(f"Conservative sleeve capacity (least-liquid active name at cap weight): "
          f"~{aums[0]/1e6:,.0f}-{np.median(aums)/1e6:,.0f} M RUB while P0 slippage holds.")
    print("\nRead: below this AUM, realistic slippage (P0) stays valid and the edge holds. Above it,")
    print("impact climbs toward breakeven (~34 bps/side); cap exposure to the least-liquid active names,")
    print("widen execution, or drop the thinnest names. Ballpark — calibrate K with live fills. is_production=false.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
