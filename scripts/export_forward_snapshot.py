"""Export an IMMUTABLE forward-accrual snapshot from the VDS — read-only, leak-safe.

Captures the evidence the paper agent accumulates while it runs the daily cycle through the
dividend season, so it can be pulled to a local machine for inspection WITHOUT touching the
production store and WITHOUT becoming a tuning input:

  state.sqlite (consistent online backup)   · data/agent/shadow_pnl.jsonl (forward-shadow track)
  data/reports/*.json + h9_shadow_pnl.txt   · the dividend feed + announcements + realised dividends
  data/news/edisclosure/*.parquet           (raw disclosure titles, so the no-lookahead verifier
                                              can re-check the feed independently on import)

A manifest records: created_at, git commit, host, the shadow-track as_of range, and a SHA256 per
file, plus an explicit "forward-accrual, read-only, NOT for tuning H9" banner (using a forward
snapshot to select/tune H9 would burn the forward gate — invariant: the forward period is
measured, never optimised on).

Properties: reads only (state.sqlite opened mode=ro; everything else copied, never modified);
writes ONLY under --out (never data/agent or data/raw); idempotent (same data -> same file
checksums); prints the manifest JSON to stdout. Import side: scripts/import_forward_snapshot.py.

    python scripts/export_forward_snapshot.py                       # -> data/forward_exports/<UTCdate>/
    python scripts/export_forward_snapshot.py --out /tmp/fwd_snap   # then scp the dir to local
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import socket
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

SCHEMA_VERSION = 1
NOT_FOR_TUNING = ("forward-accrual evidence — READ-ONLY. Do NOT use to tune or select H9 "
                  "(optimising on the forward period burns the forward gate). Measurement only.")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT),
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def _backup_sqlite_ro(src: Path, dest: Path) -> bool:
    """Consistent online backup of a (possibly WAL) SQLite DB, source opened READ-ONLY."""
    if not src.exists():
        return False
    src_conn = sqlite3.connect(f"file:{src.as_posix()}?mode=ro", uri=True)
    try:
        dst_conn = sqlite3.connect(str(dest))
        try:
            src_conn.backup(dst_conn)
        finally:
            dst_conn.close()
    finally:
        src_conn.close()
    return True


def _shadow_as_of_range(shadow_log: Path) -> dict:
    """min/max as_of + cycle count from the forward-shadow track (for the manifest)."""
    if not shadow_log.exists():
        return {"min": None, "max": None, "n_cycles": 0}
    as_ofs, n = [], 0
    for line in shadow_log.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        n += 1
        if rec.get("as_of"):
            as_ofs.append(str(rec["as_of"]))
    as_ofs.sort()
    return {"min": as_ofs[0] if as_ofs else None, "max": as_ofs[-1] if as_ofs else None,
            "n_cycles": n}


def _copy(src: Path, dest: Path) -> bool:
    if not src.exists():
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return True


def export(out_dir: Path, config_path: str | None) -> dict:
    from agent.src.config import load_config

    cfg = load_config(config_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    # self-ignore so a snapshot under data/ is never committed — but NEVER clobber an existing
    # .gitignore: a custom --out one level under data/ would otherwise overwrite data/.gitignore
    # with "*" and ignore the whole tree.
    parent_ignore = out_dir.parent / ".gitignore"
    if not parent_ignore.exists():
        parent_ignore.write_text("*\n!.gitignore\n", encoding="utf-8")

    captured: list[tuple[str, Path]] = []     # (relpath in snapshot, source-or-dest)
    missing: list[str] = []

    # 1. state.sqlite — consistent read-only backup
    state_dest = out_dir / "state.sqlite"
    if _backup_sqlite_ro(Path(cfg.state_db), state_dest):
        captured.append(("state.sqlite", state_dest))
    else:
        missing.append("state.sqlite")

    # 2. forward-shadow track
    if _copy(Path(cfg.shadow_log), out_dir / "shadow_pnl.jsonl"):
        captured.append(("shadow_pnl.jsonl", out_dir / "shadow_pnl.jsonl"))
    else:
        missing.append("shadow_pnl.jsonl")

    # 3. reports — every *.json + the realised shadow-gate verdict
    reports = REPO_ROOT / "data" / "reports"
    for src in sorted(reports.glob("*.json")):
        if _copy(src, out_dir / "reports" / src.name):
            captured.append((f"reports/{src.name}", out_dir / "reports" / src.name))
    if _copy(reports / "h9_shadow_pnl.txt", out_dir / "reports" / "h9_shadow_pnl.txt"):
        captured.append(("reports/h9_shadow_pnl.txt", out_dir / "reports" / "h9_shadow_pnl.txt"))

    # 4. dividend feed + realised events + raw disclosure titles (for the import-side verifier)
    news = REPO_ROOT / "data" / "news"
    for rel, src in (("feed/dividend_calendar_upcoming.csv", news / "dividend_calendar_upcoming.csv"),
                     ("feed/dividend_announcements.csv", news / "dividend_announcements.csv"),
                     ("feed/dividends.csv", REPO_ROOT / "data" / "raw" / "dividends.csv")):
        if _copy(src, out_dir / rel):
            captured.append((rel, out_dir / rel))
        else:
            missing.append(rel)
    for src in sorted((news / "edisclosure").glob("*.parquet")):
        rel = f"feed/edisclosure/{src.name}"
        if _copy(src, out_dir / rel):
            captured.append((rel, out_dir / rel))
    if not any(rel.startswith("feed/edisclosure/") for rel, _ in captured):
        # no raw disclosure titles bundled -> the import verifier can only SKIP the independent
        # feed no-lookahead re-check. Record the gap explicitly so it is visible in the manifest
        # (a silent skip must never read as "verified").
        missing.append("feed/edisclosure/*.parquet (feed no-lookahead NOT independently re-verifiable)")

    files = [{"name": rel, "bytes": dest.stat().st_size, "sha256": _sha256(dest)}
             for rel, dest in captured]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "snapshot_type": "forward_accrual",
        "read_only": True,
        "not_for_tuning": NOT_FOR_TUNING,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "host": socket.gethostname(),
        "agent_mode": cfg.mode,
        "as_of_range": _shadow_as_of_range(Path(cfg.shadow_log)),
        "files": files,
        "missing": missing,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                           encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    default_out = REPO_ROOT / "data" / "forward_exports" / datetime.now(timezone.utc).strftime("%Y-%m-%d")
    ap.add_argument("--out", default=str(default_out), help="snapshot output dir (NOT the prod store)")
    ap.add_argument("--config", default=None, help="agent config path (paths.state_db etc.)")
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    # refuse to write into the live store, ever
    for guard in (REPO_ROOT / "data" / "agent", REPO_ROOT / "data" / "raw"):
        if guard.resolve() in out_dir.resolve().parents or out_dir.resolve() == guard.resolve():
            print(f"refusing --out inside the production store: {out_dir}", file=sys.stderr)
            return 2

    manifest = export(out_dir, args.config)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
