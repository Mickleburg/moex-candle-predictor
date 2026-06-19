"""H9 step 3 — REALIZED-P&L shadow gate: the measurable criterion to lift is_production=false.

The H9 edge is validated on history (IS) but the FORWARD 2025+ slice is thin. The honest gate
(per project methodology: deployment-sim on a FRESH forward) is: do the dividend events that have
ACTUALLY closed on the forward realize the same market-adjusted run-up the in-sample distribution
showed? This script measures that — not positions held (that's the monitor's shadow_log), but the
REALIZED per-event return under the DEPLOYED rule.

Method (identical to the validated research so the gate measures the same thing):
  - anchor = last trading day <= RECORD date; enter -ENTRY_OFFSET TD, exit -EXIT_OFFSET TD (before the
    ex-gap); per-event return = sum of market-adjusted (stock - IMOEX) daily returns over the hold,
    net of round-trip fees. Same `runup_capture` convention as `h9_dividend_research.py`.
  - merged calendar (history + LLM forward feed) so upcoming events show as the PENDING pipeline.
  - split IN-SAMPLE (<2025, the benchmark) vs FORWARD (>=2025, the shadow track). A FUTURE-dated event
    (record date beyond the price panel) is PENDING — it has no realized number yet.

Gate verdict (honest): forward realizes net>0, %pos>0.5, dose-response holds, AND enough events have
accrued (FWD_GATE_MIN_EVENTS). Until then: ACCRUING — keep running through dividend seasons.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ML_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = ML_DIR.parent
sys.path.insert(0, str(ML_DIR))

from scripts.h9_dividend_research import load_daily, runup_capture, FEE_RT, UNIVERSE  # noqa: E402
from src.service.dividend_sleeve import load_dividend_calendar, ENTRY_OFFSET, EXIT_OFFSET  # noqa: E402

FORWARD_START = pd.Timestamp("2025-01-01", tz="Europe/Moscow")
FWD_GATE_MIN_EVENTS = 25     # forward events needed before the gate can be MET (IS had ~250)
OUT = REPO_ROOT / "data" / "reports" / "h9_shadow_pnl.txt"
SHADOW_LOG = REPO_ROOT / "data" / "reports" / "dividend_shadow_log.csv"


def capture_events(closes: dict, imoex, cal: pd.DataFrame, entry: int, exit_off: int) -> pd.DataFrame:
    """Per-event realized run-up with CLOSED/PENDING status. CLOSED = full -entry..-exit window is in
    the past (record date <= last price); PENDING = future-dated event (no realized number yet)."""
    rows = []
    for _, row in cal.iterrows():
        t = row["ticker"]
        if t not in closes:
            continue
        s = closes[t]
        rec = row["date"]
        if rec > s.index[-1]:                       # future event -> pipeline, not yet realized
            rows.append({"ticker": t, "date": rec, "year": int(rec.year), "yield": np.nan,
                         "runup": np.nan, "status": "pending"})
            continue
        pos = s.index.searchsorted(rec, side="right") - 1
        if pos < abs(entry) + 2 or pos >= len(s) - 3:   # not enough history around the anchor
            continue
        idx = s.index[pos + entry: pos + exit_off + 1]
        ar = (s.reindex(idx).pct_change() - imoex.reindex(idx).pct_change()).iloc[1:].sum()
        if not np.isfinite(ar):
            continue
        rows.append({"ticker": t, "date": rec, "year": int(rec.year),
                     "yield": float(row["value"]) / float(s.iloc[pos]),
                     "runup": float(ar), "status": "closed"})
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, label: str, pr) -> dict:
    """Net-of-fee summary of a CLOSED-event set; returns key stats for the gate."""
    c = df[df["status"] == "closed"]
    if len(c) == 0:
        pr(f"  {label:18}: no closed events")
        return {"n": 0, "net": 0.0, "pos": 0.0, "hi_net": 0.0, "dose_ok": False}
    net = c["runup"].mean() - FEE_RT
    pos = float((c["runup"] > 0).mean())
    med = c["runup"].median()
    hi = c[c["yield"] >= c["yield"].median()]
    lo = c[c["yield"] < c["yield"].median()]
    hi_net = hi["runup"].mean() - FEE_RT
    lo_net = lo["runup"].mean() - FEE_RT
    pr(f"  {label:18}: n={len(c):>3}  net {net:+.4f}  %pos {pos:.2f}  median {med:+.4f}  "
       f"| high-yield net {hi_net:+.4f} vs low {lo_net:+.4f}")
    return {"n": len(c), "net": net, "pos": pos, "hi_net": hi_net, "dose_ok": hi_net > lo_net}


def main() -> int:
    lines: list[str] = []

    def pr(s: str = "") -> None:
        print(s)
        lines.append(s)

    closes = {t: load_daily(t) for t in UNIVERSE}
    closes = {t: s for t, s in closes.items() if s is not None}
    imoex = load_daily("IMOEX")
    cal = load_dividend_calendar()
    last_price = max(s.index[-1] for s in closes.values())

    pr("H9 realized-P&L SHADOW GATE — does the forward realize the in-sample run-up?")
    pr(f"  deployed rule: enter -{ENTRY_OFFSET} TD / exit -{EXIT_OFFSET} TD vs RECORD date, "
       f"market-adjusted (stock - IMOEX), net round-trip fee {FEE_RT:.4f}")
    pr(f"  price panel through {last_price.date()}; calendar = history + LLM forward feed")
    pr("=" * 80)

    ev = capture_events(closes, imoex, cal, -ENTRY_OFFSET, -EXIT_OFFSET)
    closed = ev[ev["status"] == "closed"]
    pending = ev[ev["status"] == "pending"]
    is_ev = closed[closed["date"] < FORWARD_START]
    fw_ev = closed[closed["date"] >= FORWARD_START]

    # cross-check: our closed-event mean must reproduce the research runup_capture (methodology identity)
    ref = runup_capture(closes, imoex, cal.rename(columns={}), -ENTRY_OFFSET, -EXIT_OFFSET)
    match = abs(ref["runup"].mean() - closed["runup"].mean()) < 1e-9
    pr(f"\n[methodology identity] vs research runup_capture: closed mean {closed['runup'].mean():+.5f} "
       f"== ref {ref['runup'].mean():+.5f}  -> {'OK' if match else 'MISMATCH'}")

    pr("\nBENCHMARK (in-sample <2025) — what the forward must match:")
    is_stats = summarize(is_ev, "IN-SAMPLE", pr)

    pr("\nSHADOW TRACK (forward >=2025) — realized so far:")
    fw_stats = summarize(fw_ev, "FORWARD", pr)
    if len(fw_ev):
        pr("  per-year (forward):")
        for y, g in fw_ev.groupby("year"):
            net = g["runup"].mean() - FEE_RT
            pr(f"    {y}: n={len(g):>2}  net {net:+.4f}  %pos {(g['runup']>0).mean():.2f}")
        pr("  forward closed events (ticker / record / yield / net runup):")
        for _, r in fw_ev.sort_values("date").iterrows():
            pr(f"    {r['ticker']:5} {r['date'].date()}  yld {r['yield']:.2%}  "
               f"net {r['runup']-FEE_RT:+.4f}")

    # placebo band (in-sample) to place the forward mean against
    pr("\nPLACEBO band (random non-dividend dates, in-sample) — forward should sit in its right tail:")
    pr(_placebo_band(closes, imoex, cal, is_ev, fw_stats, -ENTRY_OFFSET, -EXIT_OFFSET))

    # pending pipeline (upcoming, not yet realized)
    pr(f"\nPENDING pipeline ({len(pending)} upcoming events, awaiting realization):")
    for _, r in pending.sort_values("date").iterrows():
        td = int(np.busday_count(last_price.date(), r["date"].date()))
        pr(f"    {r['ticker']:5} record {r['date'].date()}  (~{td} TD ahead)")

    # live-monitor reconciliation
    if SHADOW_LOG.exists():
        log = pd.read_csv(SHADOW_LOG)
        held = log[log["n_holding"].astype(int) > 0]
        pr(f"\nLive monitor shadow_log: {len(log)} runs, {len(held)} with holdings "
           f"(accrues real-time as July-2026 events trade).")

    # --- GATE VERDICT --------------------------------------------------------------------------------
    n_fw = fw_stats["n"]
    edge_ok = fw_stats["net"] > 0 and fw_stats["pos"] > 0.5
    enough = n_fw >= FWD_GATE_MIN_EVENTS
    pr("\n" + "=" * 80)
    pr(f"benchmark vs shadow: IS net {is_stats['net']:+.4f} (dose-resp {'OK' if is_stats['dose_ok'] else 'NO'}, "
       f"n={is_stats['n']}) | FWD net {fw_stats['net']:+.4f} (dose-resp "
       f"{'OK' if fw_stats['dose_ok'] else 'INVERTED'}, n={n_fw})")
    if enough and edge_ok:
        verdict = (f"GATE MET (pending sign-off): {n_fw} forward events, net {fw_stats['net']:+.4f}, "
                   f"%pos {fw_stats['pos']:.2f} — forward realizes the edge. Recommend team sign-off "
                   f"to lift is_production=false.")
    elif edge_ok:
        verdict = (f"ACCRUING (sign is RIGHT, sample THIN): {n_fw}/{FWD_GATE_MIN_EVENTS} forward events, "
                   f"net {fw_stats['net']:+.4f}, %pos {fw_stats['pos']:.2f}. Keep running through seasons "
                   f"(July-2026 pipeline = {len(pending)} events). is_production stays false.")
    else:
        verdict = (f"NOT MET: {n_fw} forward events, net {fw_stats['net']:+.4f}, %pos {fw_stats['pos']:.2f} "
                   f"— forward does NOT confirm the edge yet. is_production stays false; do not deploy live.")
    pr("VERDICT: " + verdict)
    pr("  (is_production=false until this gate is MET on accrued forward events AND team sign-off.)")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n-> {OUT}")
    return 0


def _placebo_band(closes, imoex, cal, is_ev, fw_stats, entry, exit_off, n_trials=200) -> str:
    """In-sample placebo distribution of the capture on random non-dividend dates; compare forward."""
    def cap_at(s, pos):
        idx = s.index[pos + entry: pos + exit_off + 1]
        ar = (s.reindex(idx).pct_change() - imoex.reindex(idx).pct_change()).iloc[1:].sum()
        return ar if np.isfinite(ar) else np.nan
    expos: dict = {}
    for _, row in cal.iterrows():
        t = row["ticker"]
        if t not in closes or row["date"] >= FORWARD_START or row["date"] > closes[t].index[-1]:
            continue
        s = closes[t]; pos = s.index.searchsorted(row["date"], side="right") - 1
        if abs(entry) + 2 <= pos < len(s) - 3:
            expos.setdefault(t, set()).add(pos)
    rng = np.random.default_rng(0); pmeans = []
    for _ in range(n_trials):
        vals = []
        for t, ev in expos.items():
            s = closes[t]
            valid = [p for p in range(abs(entry) + 2, len(s) - 3) if all(abs(p - e) > 20 for e in ev)]
            if not valid:
                continue
            for p in rng.choice(valid, size=min(len(ev), len(valid)), replace=False):
                v = cap_at(s, p)
                if np.isfinite(v):
                    vals.append(v)
        if vals:
            pmeans.append(np.mean(vals))
    pmeans = np.array(pmeans)
    fw_gross = fw_stats["net"] + FEE_RT
    pctl = float((pmeans < fw_gross).mean()) if len(pmeans) else float("nan")
    return (f"  placebo mean {pmeans.mean():+.4f} [2.5-97.5%: {np.percentile(pmeans,2.5):+.4f}.."
            f"{np.percentile(pmeans,97.5):+.4f}]; forward gross {fw_gross:+.4f} at "
            f"{pctl:.0%} of placebo  -> {'in right tail (edge-like)' if pctl>0.95 else 'NOT yet separated (thin)'}")


if __name__ == "__main__":
    raise SystemExit(main())
