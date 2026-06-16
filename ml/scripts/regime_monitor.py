"""Shadow monitor for the regime-gate defensive product.

Run daily: computes the current market-regime state (past-only) and prints a defensive
recommendation, appending a row to a shadow log so a live track record accrues. This is the
paper/shadow mechanism for the gate — no orders, just the exposure recommendation over time.

    python ml/scripts/regime_monitor.py
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ML_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = ML_DIR.parent
sys.path.insert(0, str(ML_DIR))

from src.features.cross_sectional import load_panels  # noqa: E402
from src.service.risk_analytics import regime_state  # noqa: E402

LOG = REPO_ROOT / "data" / "reports" / "regime_shadow_log.csv"


def main() -> int:
    panel, _, market = load_panels(timeframe="1D")
    as_of = panel.index[-1]
    st = regime_state(panel, market, as_of)
    state = "NOVEL — REDUCE EXPOSURE" if st["novel"] else "NORMAL — full exposure"
    print(f"Regime monitor @ {as_of.date()}")
    print(f"  novelty distance = {st['distance']}  (percentile {st['percentile']:.0%})")
    print(f"  state: {state}")
    print(f"  recommended gross-exposure scalar = {st['exposure_scalar']:.2f}")

    row = {"run_at": datetime.now().isoformat(timespec="seconds"),
           "as_of": as_of.isoformat(), "distance": st["distance"],
           "percentile": st["percentile"], "novel": st["novel"],
           "exposure_scalar": st["exposure_scalar"]}
    LOG.parent.mkdir(parents=True, exist_ok=True)
    hist = pd.read_csv(LOG) if LOG.exists() else pd.DataFrame()
    hist = pd.concat([hist, pd.DataFrame([row])], ignore_index=True)
    hist = hist.drop_duplicates(subset=["as_of"], keep="last")
    hist.to_csv(LOG, index=False)
    print(f"  logged -> {LOG} ({len(hist)} entries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
