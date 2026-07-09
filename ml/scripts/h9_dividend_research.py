"""H9 — Dividend / calendar effect (different axis: event/calendar, not ranking/reversion).

THESIS
    MOEX has very high dividend yields and heavy retail participation. Two documented anomalies
    worth testing: (a) pre-ex-date RUN-UP (price drifts up into the ex-date as buyers chase the
    dividend), (b) post-ex-date DRIFT (the ex-day gap-down over/under-shoots and reverts). Either,
    if real and stable, is a calendar edge independent of cross-sectional ranking (H1/H2/H6) and
    pairwise reversion (H7), all of which failed.

METHOD (careful, two stages)
    Stage 1 (this file, DESCRIPTIVE, no trading): event study of MARKET-ADJUSTED returns
    (stock - IMOEX) in a window around each ex-dividend event. Average abnormal return (AAR) and
    cumulative (CAAR), split IN-SAMPLE (<2025) vs FORWARD (>=2025) and by dividend-yield bucket.
    The ex-day is located EMPIRICALLY (the biggest average drop) rather than assumed, since the
    record date (ISS registryclosedate) is ~1-2 trading days after the ex-date under T+1/T+2.
    Stage 2 (only if a pattern exists): a tradeable rule + deployment-sim with fees, IS vs forward.

NO-LOOKAHEAD NOTE
    The descriptive study is not a strategy (no trading), so no lookahead. A later tradeable rule
    would rely on the ex-date being KNOWN ahead of entry — true on MOEX (board recommends, AGM
    approves dividends weeks before the record date), so entering N days pre-ex is realistic.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ML_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = ML_DIR.parent
sys.path.insert(0, str(ML_DIR))

from src.data.load import to_moscow_time  # noqa: E402
from src.service.dividend_universe import (  # noqa: E402
    active_universe, resolve_universe, FORWARD_START,
)

DATA_RAW = REPO_ROOT / "data" / "raw"
# Single-source universe (src/service/dividend_universe.py). Module-level default keeps every importer
# (gate, no-lookahead, sim, ...) working; env H9_UNIVERSE=expanded or main()'s --universe flips it.
UNIVERSE = active_universe()
WIN = 15  # trading-day window each side of the event anchor


def load_daily(ticker: str) -> pd.Series | None:
    files = sorted(DATA_RAW.glob(f"{ticker}_1H_*.parquet"))
    if not files:
        return None
    df = pd.read_parquet(files[-1]); df.columns = [c.lower() for c in df.columns]
    s = pd.Series(df["close"].to_numpy(float), index=to_moscow_time(df["begin"]))
    return s[~s.index.duplicated(keep="last")].sort_index().resample("1D").last().dropna()


FEE_RT = 0.0005 * 2  # round-trip fee (open+close, one-way 5bps)


def runup_capture(closes, imoex, div, entry: int, exit_off: int) -> pd.DataFrame:
    """Per-event market-adjusted return from `entry` to `exit_off` (trading-day offsets vs the
    record-date anchor). exit_off must be <= -2 to sell BEFORE the ex-day gap (located at ~-1)."""
    recs = []
    for _, row in div.iterrows():
        t = row["ticker"]
        if t not in closes:
            continue
        s = closes[t]
        if row["date"] > s.index[-1]:          # future-dated (beyond panel) -> not realized, skip
            continue
        pos = s.index.searchsorted(row["date"], side="right") - 1
        # window = bars [pos+entry .. pos+exit_off]; exit_off<0 sells BEFORE the record, so NO bars
        # after the record are needed. Require only both offsets in-range. (Was `pos >= len(s)-3`,
        # needing 3 FUTURE bars -> false-dropped the most-recent realized events, undercounting n.)
        if pos + entry < 0 or not (0 <= pos + exit_off < len(s)):
            continue
        idx = s.index[pos + entry: pos + exit_off + 1]
        ar = (s.reindex(idx).pct_change() - imoex.reindex(idx).pct_change()).iloc[1:].sum()
        if np.isfinite(ar):
            recs.append({"ticker": t, "year": int(row["date"].year),
                         "yield": row["value"] / s.iloc[pos], "runup": float(ar)})
    return pd.DataFrame(recs)


def stage2(closes, imoex, div) -> None:
    print("\n=== STAGE 2: dividend RUN-UP capture (buy pre-ex, sell before ex-gap, market-hedged) ===")
    r = runup_capture(closes, imoex, div, entry=-10, exit_off=-2)
    print(f"base (entry -10, exit -2), {len(r)} events, round-trip fee {FEE_RT:.4f}:")
    print(f"  OVERALL mean {r.runup.mean():+.4f} net {r.runup.mean()-FEE_RT:+.4f} "
          f"%pos {(r.runup>0).mean():.2f} median {r.runup.median():+.4f}")
    g = r.groupby("year").agg(n=("runup", "size"), mean=("runup", "mean"),
                              pos=("runup", lambda x: (x > 0).mean()))
    print("  per-year:")
    for y, rr in g.iterrows():
        print(f"    {y}: n={int(rr['n']):>2} mean {rr['mean']:+.4f} %pos {rr['pos']:.2f}")
    hi = r[r["yield"] >= r["yield"].median()]
    print(f"  DOSE-RESPONSE high-yield half: net {hi.runup.mean()-FEE_RT:+.4f} "
          f"%pos {(hi.runup>0).mean():.2f}  (vs low-yield net "
          f"{r[r['yield']<r['yield'].median()].runup.mean()-FEE_RT:+.4f}) -> scales with yield = real")
    print("  ENTRY-offset sensitivity (exit -2), net of fees:")
    for e in (-15, -12, -10, -8, -6, -4):
        rr = runup_capture(closes, imoex, div, e, -2)
        print(f"    entry {e:>3}: net {rr.runup.mean()-FEE_RT:+.4f} %pos {(rr.runup>0).mean():.2f}")
    print("  EXIT-offset sensitivity (entry -10): exit <=-2 avoids the ex-gap at ~-1")
    for x in (-3, -2, -1, 0):
        rr = runup_capture(closes, imoex, div, -10, x)
        print(f"    exit {x:>3}: net {rr.runup.mean()-FEE_RT:+.4f} %pos {(rr.runup>0).mean():.2f}")
    print("  PLACEBO control (is it dividend-specific or just momentum?):")
    placebo_test(closes, imoex, div)
    print("  Read: positive & stable across entry windows (-15..-8) AND IS years AND scaling with")
    print("  yield AND distinct from a random-date placebo = a robust, dividend-specific calendar edge.")
    print("  Caveat: 2025 forward is thin (few events) - accrue more; verify no-lookahead vs announce dates.")


def placebo_test(closes, imoex, div, entry: int = -10, exit_off: int = -2, n_trials: int = 200) -> None:
    """Control: run the SAME capture on RANDOM non-dividend dates (>20d from any event). If the
    run-up appears there too, it's generic momentum, not a dividend effect. Real should sit far in
    the right tail of the placebo distribution."""
    def cap_at(s, pos):
        idx = s.index[pos + entry: pos + exit_off + 1]
        ar = (s.reindex(idx).pct_change() - imoex.reindex(idx).pct_change()).iloc[1:].sum()
        return ar if np.isfinite(ar) else np.nan
    real, expos = [], {}
    for _, row in div.iterrows():
        t = row["ticker"]
        if t not in closes:
            continue
        s = closes[t]; pos = s.index.searchsorted(row["date"], side="right") - 1
        if pos < abs(entry) + 2 or pos >= len(s) - 3:
            continue
        real.append(cap_at(s, pos)); expos.setdefault(t, set()).add(pos)
    real = np.array([x for x in real if np.isfinite(x)])
    rng = np.random.default_rng(0); pmeans = []
    for _ in range(n_trials):
        vals = []
        for t, s in closes.items():
            ev = expos.get(t, set())
            if not ev:
                continue
            valid = [p for p in range(abs(entry) + 2, len(s) - 3) if all(abs(p - e) > 20 for e in ev)]
            for p in rng.choice(valid, size=min(len(ev), len(valid)), replace=False):
                v = cap_at(s, p)
                if np.isfinite(v):
                    vals.append(v)
        pmeans.append(np.mean(vals))
    pmeans = np.array(pmeans)
    z = (real.mean() - pmeans.mean()) / pmeans.std()
    print(f"  PLACEBO (random non-div dates, {n_trials} trials): mean {pmeans.mean():+.4f} "
          f"[2.5-97.5%: {np.percentile(pmeans,2.5):+.4f}..{np.percentile(pmeans,97.5):+.4f}]")
    print(f"  REAL {real.mean():+.4f} vs placebo: z={z:+.2f}, p(placebo>=real)={ (pmeans>=real.mean()).mean():.3f}"
          f"  -> {'dividend-specific (not momentum)' if z>2 else 'NOT distinguishable from momentum'}")


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="H9 dividend event study + run-up capture (Stage 1+2).")
    ap.add_argument("--universe", default=None, choices=["current", "expanded"],
                    help="universe to run (default: env H9_UNIVERSE or 'current')")
    args = ap.parse_args()
    universe = resolve_universe(args.universe) if args.universe else UNIVERSE
    closes = {t: load_daily(t) for t in universe}
    closes = {t: s for t, s in closes.items() if s is not None}
    missing = [t for t in universe if t not in closes]
    print(f"universe: {len(universe)} requested, {len(closes)} with data"
          + (f" (MISSING, awaiting backend: {missing})" if missing else ""))
    imoex = load_daily("IMOEX")
    div = pd.read_csv(DATA_RAW / "dividends.csv")
    div["date"] = pd.to_datetime(div["date"]).dt.tz_localize("Europe/Moscow")
    div = div.dropna(subset=["value"])
    div = div[div["value"] > 0]

    # collect market-adjusted return windows per event
    is_mat, fw_mat, ylds = [], [], []
    n_used = 0
    for _, row in div.iterrows():
        t = row["ticker"]
        if t not in closes:
            continue
        s = closes[t]
        rdate = row["date"]
        # anchor = position of last trading day <= record date
        pos = s.index.searchsorted(rdate, side="right") - 1
        if pos < WIN or pos >= len(s) - WIN:
            continue
        win_idx = s.index[pos - WIN: pos + WIN + 1]
        sret = s.reindex(win_idx).pct_change()
        mret = imoex.reindex(win_idx).pct_change()
        ar = (sret - mret).to_numpy()                  # abnormal (market-adjusted) return
        if not np.all(np.isfinite(ar[1:])):
            continue
        price = s.iloc[pos]
        ylds.append(row["value"] / price)
        (fw_mat if rdate >= FORWARD_START else is_mat).append(ar)
        n_used += 1

    offs = np.arange(-WIN, WIN + 1)
    def caar(mat):
        if not mat:
            return None
        a = np.nanmean(np.array(mat), axis=0)
        a[0] = 0.0
        return np.cumsum(a)

    print(f"H9 dividend event study - {n_used} events used "
          f"(IS {len(is_mat)}, FWD {len(fw_mat)}); median yield {np.median(ylds):.2%}")
    print(f"window = record date +/- {WIN} trading days (ex-day is ~ -1/-2 from record under T+1/T+2)\n")
    print(f"{'offset':>6} | {'CAAR_IS':>9} {'AAR_IS':>8} | {'CAAR_FWD':>9}")
    ci, cf = caar(is_mat), caar(fw_mat)
    ai = np.nanmean(np.array(is_mat), axis=0)
    for i, o in enumerate(offs):
        mark = "  <-- record date" if o == 0 else ("  <-- ex (~)" if o in (-1, -2) else "")
        fwv = f"{cf[i]:+.4f}" if cf is not None else "   n/a"
        print(f"{o:>6} | {ci[i]:+.4f} {ai[i]:+.4f} | {fwv}{mark}")

    # quantify: pre-run-up (offset -10..-2), ex-gap (-2..0), post-drift (0..+10)
    def seg(c, lo, hi):
        return float(c[np.where(offs == hi)[0][0]] - c[np.where(offs == lo)[0][0]])
    print(f"\nIS segments: pre-runup[-10..-2] {seg(ci,-10,-2):+.4f} | "
          f"ex-gap[-2..0] {seg(ci,-2,0):+.4f} | post-drift[0..+10] {seg(ci,0,10):+.4f}")
    if cf is not None:
        print(f"FWD segments: pre-runup[-10..-2] {seg(cf,-10,-2):+.4f} | "
              f"ex-gap[-2..0] {seg(cf,-2,0):+.4f} | post-drift[0..+10] {seg(cf,0,10):+.4f}")
    print("\nRead: a tradeable calendar edge needs a CONSISTENT (IS and FWD) market-adjusted segment")
    print("(e.g. positive pre-run-up, or post-ex reversion) larger than ~fees. Else H9 is dead too.")

    stage2(closes, imoex, div)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
