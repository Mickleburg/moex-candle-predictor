"""Forward-shadow monitor for the H9 dividend run-up sleeve.

Run regularly (paper/shadow): fetches the LIVE dividend calendar from MOEX ISS, finds names whose
ex-date is in the entry window NOW (we should be holding the run-up), sizes them inverse-vol, lists
upcoming entries, and appends a snapshot to a shadow log so a forward track accrues — directly
addressing H9's only open caveat (the thin 2025 forward). No orders; the recommendation only.

    python ml/scripts/dividend_sleeve_monitor.py

Hold logic is in TRADING days to the record date (RU-holiday-aware shared backend calendar): enter
~12 TD before, exit ~2 TD before (avoid the ex-gap). Past-only vols for sizing. is_production=false.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ML_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = ML_DIR.parent
sys.path.insert(0, str(ML_DIR))

from scripts.xsec_eval_harness import load_daily_panel  # noqa: E402
from src.service.dividend_sleeve import (  # noqa: E402
    inverse_vol_weights, load_dividend_calendar, trading_days_between,
    ENTRY_OFFSET, EXIT_OFFSET, VOL_WINDOW, MAX_WEIGHT,
)

UNIVERSE = ["SBER", "GAZP", "LKOH", "GMKN", "ROSN", "NVTK", "TATN", "MGNT",
            "MTSS", "SNGS", "CHMF", "ALRS", "VTBR", "MAGN", "NLMK", "PLZL"]
LOG = REPO_ROOT / "data" / "reports" / "dividend_shadow_log.csv"


def td_to(as_of: pd.Timestamp, when: pd.Timestamp) -> int:
    """Signed trading-day count from as_of to `when`, RU-holiday-aware (shared backend calendar)."""
    return trading_days_between(as_of.date(), when.date())


def main() -> int:
    # --as-of YYYY-MM-DD demonstrates/tests the monitor on a past date; no arg = live (today). Calendar
    # = historical snapshot + the LLM forward feed (data/news/dividend_calendar_upcoming.csv), so upcoming
    # ex-dates appear ~37 TD before record (board recommendations), well before ISS publishes them.
    backtest_date = None
    if "--as-of" in sys.argv:
        backtest_date = pd.Timestamp(sys.argv[sys.argv.index("--as-of") + 1], tz="Europe/Moscow")
    as_of = backtest_date if backtest_date is not None else pd.Timestamp.now(tz="Europe/Moscow").normalize()
    panel = load_daily_panel(UNIVERSE)
    last_pos = panel.index.searchsorted(as_of, side="right") - 1 if backtest_date is not None else len(panel) - 1
    # merged calendar = ISS history + LLM forward feed (data/news/dividend_calendar_upcoming.csv) so
    # upcoming ex-dates are visible before ISS publishes the record date.
    cal = load_dividend_calendar()

    holding, upcoming = [], []
    for _, r in cal.iterrows():
        n = td_to(as_of, r["date"])             # trading days from today to the record date
        if EXIT_OFFSET < n <= ENTRY_OFFSET:     # in [3..12] -> hold the run-up now
            holding.append((r["ticker"], r["date"], n))
        elif ENTRY_OFFSET < n <= ENTRY_OFFSET + 5:   # entry coming up
            upcoming.append((r["ticker"], r["date"], n))

    names = [t for t, _, _ in holding if t in panel.columns]
    weights = inverse_vol_weights(panel, names, last_pos, VOL_WINDOW, MAX_WEIGHT) if names else {}

    print(f"Dividend run-up sleeve — shadow monitor @ {as_of.date()}")
    print(f"  entry ~{ENTRY_OFFSET} TD before ex, exit ~{EXIT_OFFSET} TD before (avoid ex-gap); "
          f"sector-hedged at book level by risk_manager. is_production=false")
    if holding:
        # dedupe by ticker for display (a name with two near ex-dates is ONE position); show nearest
        nearest: dict[str, tuple] = {}
        for t, d, n in holding:
            if t not in nearest or n < nearest[t][1]:
                nearest[t] = (d, n)
        print(f"  HOLDING NOW ({len(weights)} names, inv-vol weighted, hedge = -{sum(weights.values()):.2f} IMOEX):")
        for t in sorted(weights, key=lambda x: -weights[x]):
            d, n = nearest.get(t, (None, 0))
            extra = "" if sum(1 for h in holding if h[0] == t) == 1 else " (+more)"
            print(f"    {t:5} weight {weights[t]:.3f}  next ex-record {d.date()} (in {n} TD){extra}")
    else:
        print("  HOLDING NOW: none (no ex-dates in the entry window today)")
    if upcoming:
        print(f"  UPCOMING entries ({len(upcoming)}):")
        for t, d, n in sorted(upcoming, key=lambda x: x[2]):
            print(f"    {t:5} ex-record {d.date()} (enter in {n - ENTRY_OFFSET} TD, record in {n} TD)")

    row = {"run_at": datetime.now().isoformat(timespec="seconds"), "as_of": as_of.date().isoformat(),
           "n_holding": len(holding), "holding": ";".join(f"{t}:{weights.get(t,0):.3f}" for t, _, _ in holding),
           "n_upcoming": len(upcoming), "gross_long": round(sum(weights.values()), 4)}
    LOG.parent.mkdir(parents=True, exist_ok=True)
    hist = pd.read_csv(LOG) if LOG.exists() else pd.DataFrame()
    hist = pd.concat([hist, pd.DataFrame([row])], ignore_index=True).drop_duplicates(subset=["as_of"], keep="last")
    hist.to_csv(LOG, index=False)
    print(f"  logged -> {LOG} ({len(hist)} entries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
