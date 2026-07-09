"""H9 no-lookahead diagnostic (part B) — is the dividend KNOWN before we enter at offset -12?

The run-up strategy is only tradeable if the dividend (amount + ex-date) is public before entry.
Institutionally it is: by FZ-208 art.42 the record date is set by the AGM (record = AGM + 10..20
days) and the AGM notice carrying the board's dividend RECOMMENDATION must be published >=21 days
before the AGM -> the dividend goes public at the board recommendation, ~22-38 trading days before
the record date, well before our -12 entry.

DATA-DRIVEN CHECK (no announcement dates needed): if an announcement happened INSIDE our entry
window [-15,-2], it would print as a price JUMP (a spike in the average abnormal return on that day).
If instead the abnormal-return path is a smooth drift with no single-day spike, the information
pre-exists the window -> consistent with no-lookahead. The DEFINITIVE per-event check (announcement
date < entry date) still needs board-recommendation dates from e-disclosure (LLM chat).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ML_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_DIR))

from scripts.h9_dividend_research import load_daily, UNIVERSE, FORWARD_START  # noqa: E402

WIN = 15
DATA_RAW = ML_DIR.parent / "data" / "raw"


def main() -> int:
    closes = {t: load_daily(t) for t in UNIVERSE}
    closes = {t: s for t, s in closes.items() if s is not None}
    imoex = load_daily("IMOEX")
    div = pd.read_csv(DATA_RAW / "dividends.csv")
    div["date"] = pd.to_datetime(div["date"]).dt.tz_localize("Europe/Moscow")
    div = div.dropna(subset=["value"]); div = div[div["value"] > 0]

    mat = []
    for _, row in div.iterrows():
        t = row["ticker"]
        if t not in closes:
            continue
        s = closes[t]
        pos = s.index.searchsorted(row["date"], side="right") - 1
        if pos < WIN or pos >= len(s) - WIN:
            continue
        idx = s.index[pos - WIN: pos + WIN + 1]
        ar = (s.reindex(idx).pct_change() - imoex.reindex(idx).pct_change()).to_numpy()
        if np.all(np.isfinite(ar[1:])):
            mat.append(ar)
    mat = np.array(mat)
    offs = np.arange(-WIN, WIN + 1)
    aar = np.nanmean(mat, axis=0); aar[0] = 0.0

    def at(o):
        return float(aar[np.where(offs == o)[0][0]])
    entry_win = [o for o in offs if -15 <= o <= -3]               # days whose return we earn pre-exit
    max_abs_win = max(abs(at(o)) for o in entry_win)
    gap = abs(at(-1))
    runup = float(sum(at(o) for o in offs if -14 <= o <= -2))
    biggest_day = max(at(o) for o in entry_win)

    print(f"H9 no-lookahead diagnostic - {len(mat)} events")
    print(f"\nAAR per offset in the entry window (look for a JUMP = an announcement inside the window):")
    for o in offs:
        if -15 <= o <= 2:
            bar = "#" * int(abs(at(o)) * 1000)
            tag = "  <-- EX-GAP" if o in (-1, 0) else ""
            print(f"  {o:>3}: {at(o):+.4f} {bar}{tag}")
    print(f"\n  max |AAR| within entry window [-15,-3] = {max_abs_win:.4f}")
    print(f"  ex-gap |AAR| at offset -1          = {gap:.4f}  ({gap/max_abs_win:.1f}x the largest "
          f"in-window day)")
    print(f"  total run-up [-14..-2]             = {runup:+.4f}; biggest single in-window day "
          f"{biggest_day:+.4f} ({biggest_day/runup:.0%} of run-up)")
    smooth = max_abs_win < gap / 3 and biggest_day < 0.5 * runup
    print(f"\n  VERDICT: {'SMOOTH drift, no announcement spike inside window -> consistent with the' if smooth else 'a single-day spike exists -> inspect for an in-window announcement'}")
    print(f"  dividend being public BEFORE entry (no-lookahead). Definitive check still needs board-")
    print(f"  recommendation dates from e-disclosure (LLM chat) -> per-event: announce_date < entry_date.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
