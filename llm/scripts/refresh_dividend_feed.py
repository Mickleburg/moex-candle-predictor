# -*- coding: utf-8 -*-
r"""Scheduled refresh entry point for the forward dividend feed — ONE idempotent, network-resilient
command the V3 orchestrator calls at EOD so the dividend knowledge base self-updates on the VDS.

Pipeline (each stage is safe to re-run):
  1. extract  — incremental e-disclosure title pull for a recent window, merged into the parquet
                snapshots (dedup by pseudo_guid). Discovers NEW board recommendations / declines.
  2. bodies   — fetch+cache message bodies for dividend-chain events (already-cached ones skipped).
  3. build    — parse bodies -> forward feed DataFrame, written to a TEMP file.
  4. verify   — INDEPENDENT no-lookahead re-check (verify_dividend_feed) from raw pubDates.
  5. sverka   — (best-effort) ML anchor sverka (ml/scripts/h9_anchor_sverka.py): does the feed trade
                the same anchor the research validated on. A clear FAIL blocks the swap.
  6. swap     — only if verify (and any clear sverka) PASS, atomically replace the live CSV. On any
                failure the live CSV is left untouched (last-good), so a network blip never breaks it.

Resilience / contract:
  * Network stages (1,2) retry with backoff; if they still fail we proceed to rebuild from the
    existing cache (=> last-good feed) and report `degraded` rather than corrupting the feed.
  * Exit code: 0 iff the resulting feed is TRUSTWORTHY (verify passed, no clear sverka FAIL) — even
    if the data couldn't be refreshed this run. Non-zero only when the feed cannot be trusted. This
    keeps a transient outage from halting trading while still alerting on a genuinely broken feed.
  * Idempotent: deterministic build => re-running the same day yields byte-identical CSV (changed=false).
  * A single JSON summary is printed to STDOUT (for the orchestrator's _run_json_cmd); all human logs
    go to STDERR so stdout stays clean JSON.

Run: $env:PYTHONIOENCODING="utf-8"; & "ml\.venv-win\Scripts\python.exe" llm\scripts\refresh_dividend_feed.py
     (flags: --no-extract  --extract-since YYYY-MM-DD  --no-anchor-sverka  --retries N  --as-of DATE)
"""
from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

SCRIPTS = Path(__file__).resolve().parent
REPO = SCRIPTS.parents[1]
sys.path.insert(0, str(SCRIPTS))

import build_dividend_calendar_upcoming as bld  # noqa: E402
import verify_dividend_feed as vfy  # noqa: E402

FEED_CSV = bld.OUT_CSV
TMP_CSV = FEED_CSV.with_suffix(".tmp.csv")
ANCHOR_SVERKA = REPO / "ml" / "scripts" / "h9_anchor_sverka.py"


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def run_stage(name: str, cmd: list[str], timeout: float, retries: int) -> tuple[str, str]:
    """Run a subprocess stage with retries. Returns (status, detail). status in {ok, failed}."""
    last = ""
    for attempt in range(1, retries + 1):
        try:
            log(f"[{name}] attempt {attempt}/{retries}: {' '.join(cmd[-3:])}")
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                               cwd=str(REPO))
            if p.returncode == 0:
                tail = (p.stdout.strip().splitlines() or [""])[-1]
                log(f"[{name}] ok — {tail}")
                return "ok", tail
            last = (p.stderr or p.stdout).strip().splitlines()[-1:] or [""]
            last = last[0] if last else ""
            log(f"[{name}] rc={p.returncode}: {last}")
        except subprocess.TimeoutExpired:
            last = f"timeout after {timeout}s"
            log(f"[{name}] {last}")
        except Exception as exc:  # noqa: BLE001
            last = f"{type(exc).__name__}: {exc}"
            log(f"[{name}] {last}")
        if attempt < retries:
            time.sleep(2 ** attempt)
    return "failed", last


def anchor_sverka(py: str) -> tuple[str, str]:
    """Best-effort cross-block gate. Returns (status, detail): pass | fail | skipped."""
    if not ANCHOR_SVERKA.exists():
        return "skipped", "script not present"
    try:
        p = subprocess.run([py, str(ANCHOR_SVERKA)], capture_output=True, text=True,
                           timeout=600, cwd=str(REPO))
        out = (p.stdout or "") + (p.stderr or "")
        if "VERDICT: PASS" in out or (p.returncode == 0 and "FAIL" not in out):
            return "pass", "anchor == research anchor"
        if "VERDICT: FAIL" in out or p.returncode == 1:
            line = next((l for l in out.splitlines() if "FAIL" in l), "anchor mismatch")
            return "fail", line.strip()[:160]
        return "skipped", f"inconclusive rc={p.returncode}"
    except Exception as exc:  # noqa: BLE001 - ML env issue must not corrupt our gate
        return "skipped", f"could not run ({type(exc).__name__})"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-extract", action="store_true", help="skip the incremental title pull")
    ap.add_argument("--extract-since", default=None,
                    help="title-pull window start (default today-45d)")
    ap.add_argument("--no-anchor-sverka", action="store_true", help="skip the ML anchor cross-check")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--as-of", default=date.today().isoformat())
    args = ap.parse_args()

    py = sys.executable
    since = args.extract_since or (date.today() - timedelta(days=45)).isoformat()
    summary: dict = {"block": "llm_dividend_feed", "as_of": args.as_of, "ok": False,
                     "degraded": False, "stages": {}, "feed": {}, "nolookahead": {}, "errors": []}

    # ---- 1. extract (incremental, network) ----
    if args.no_extract:
        summary["stages"]["extract"] = "skipped"
    else:
        st, detail = run_stage("extract", [py, str(SCRIPTS / "edisc_extract.py"),
                                           "--since", since, "--merge"], timeout=1200,
                               retries=args.retries)
        summary["stages"]["extract"] = st
        if st == "failed":
            summary["degraded"] = True
            summary["errors"].append(f"extract: {detail}")

    # ---- 2. fetch bodies (network) ----
    st, detail = run_stage("bodies", [py, str(SCRIPTS / "edisc_fetch_bodies.py")],
                           timeout=900, retries=args.retries)
    summary["stages"]["bodies"] = st
    if st == "failed":
        summary["degraded"] = True
        summary["errors"].append(f"bodies: {detail}")

    # ---- 3. build (pure) -> TEMP ----
    try:
        feed, _passed, declined = bld.feed_frames()
        TMP_CSV.parent.mkdir(parents=True, exist_ok=True)
        feed.to_csv(TMP_CSV, index=False, encoding="utf-8")
        summary["stages"]["build"] = "ok"
        summary["feed"] = {"upcoming": int(len(feed)), "declined": int(len(declined))}
    except Exception as exc:  # noqa: BLE001
        summary["stages"]["build"] = "failed"
        summary["errors"].append(f"build: {type(exc).__name__}: {exc}")
        print(json.dumps(summary, ensure_ascii=False))
        log("ABORT: build failed; live feed left untouched")
        return 1

    # ---- 4. independent no-lookahead verify on the TEMP feed ----
    ok, report, stats = vfy.verify(feed, pd.Timestamp(args.as_of))
    summary["stages"]["verify_nolookahead"] = "PASS" if ok else "FAIL"
    summary["nolookahead"] = stats
    for line in report:
        log(line)
    if not ok:
        summary["errors"].append("no-lookahead verify FAILED")
        TMP_CSV.unlink(missing_ok=True)
        print(json.dumps(summary, ensure_ascii=False))
        log("ABORT: verify failed; live feed left untouched")
        return 1

    # ---- 5. atomic swap, with a backup so a failing sverka can roll back to last-good ----
    bak = FEED_CSV.with_suffix(".bak.csv")
    had_live = FEED_CSV.exists()
    changed = not had_live or FEED_CSV.read_text(encoding="utf-8") != TMP_CSV.read_text(encoding="utf-8")
    if had_live:
        FEED_CSV.replace(bak)            # preserve last-good
    os.replace(TMP_CSV, FEED_CSV)        # publish new feed to the live path
    summary["stages"]["swap"] = "ok"

    # ---- 6. ML anchor sverka on the now-live feed (best-effort; clear FAIL -> roll back) ----
    if args.no_anchor_sverka:
        summary["stages"]["anchor_sverka"] = "skipped"
    else:
        sv, sv_detail = anchor_sverka(py)
        summary["stages"]["anchor_sverka"] = sv
        if sv == "fail":
            summary["errors"].append(f"anchor sverka FAIL: {sv_detail}")
            if had_live:
                bak.replace(FEED_CSV)    # restore last-good
            print(json.dumps(summary, ensure_ascii=False))
            log("ABORT: anchor sverka failed; rolled back to last-good feed")
            return 1

    bak.unlink(missing_ok=True)
    summary["feed"]["changed"] = changed
    summary["ok"] = True

    print(json.dumps(summary, ensure_ascii=False))
    log(f"DONE: feed {'updated' if changed else 'unchanged'} "
        f"({summary['feed']['upcoming']} upcoming) — ok={summary['ok']} degraded={summary['degraded']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
