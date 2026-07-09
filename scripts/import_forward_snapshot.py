"""Import + VERIFY a forward-accrual snapshot locally (read-only landing).

Counterpart to scripts/export_forward_snapshot.py. Takes an exported snapshot dir (scp'd from the
VDS) and, only if every check passes, lands it under data/forward_snapshots/<date>/ as READ-ONLY
evidence. Fails loudly (non-zero exit) on ANY discrepancy.

Checks:
  1. SHA256 — every file in the manifest is present and matches its recorded hash (integrity /
     tamper / truncated-transfer detection).
  2. Shadow-track no-lookahead + monotonicity — in shadow_pnl.jsonl: as_of is non-decreasing
     (no rewind), each record's trade_date <= its as_of (decision uses only past), no duplicate
     trade_date.
  3. Dividend-feed no-lookahead — REUSES the independent verifier llm/scripts/verify_dividend_feed
     (re-checks board_reco_date <= as_of from the bundled raw disclosure parquet). Skipped with a
     warning only if its inputs/deps are absent — never silently passed.

The landed copy is chmod read-only; data/forward_snapshots/ self-ignores (a `.gitignore` of `*`),
so forward evidence is never committed and never edited in place (it is measurement, not a tuning
input — see the snapshot manifest's not_for_tuning banner).

    python scripts/import_forward_snapshot.py ./incoming_snapshot        # -> data/forward_snapshots/<date>/
    python scripts/import_forward_snapshot.py ./incoming_snapshot --force --json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOTS_ROOT = REPO_ROOT / "data" / "forward_snapshots"


class VerifyError(Exception):
    """Raised on any discrepancy — import must fail."""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _check_sha256(snapshot: Path, manifest: dict) -> list[str]:
    problems: list[str] = []
    for entry in manifest.get("files", []):
        f = snapshot / entry["name"]
        if not f.exists():
            problems.append(f"missing file {entry['name']}")
            continue
        actual = _sha256(f)
        if actual != entry["sha256"]:
            problems.append(f"sha256 mismatch {entry['name']}: {actual[:12]} != {entry['sha256'][:12]}")
    return problems


def _parse_as_of(value: str) -> datetime:
    return datetime.fromisoformat(str(value))


def _check_shadow_track(snapshot: Path) -> list[str]:
    """as_of monotonic non-decreasing; trade_date <= as_of; no duplicate trade_date."""
    log = snapshot / "shadow_pnl.jsonl"
    if not log.exists():
        return []   # nothing accrued yet is not a violation
    problems: list[str] = []
    prev_as_of: datetime | None = None
    seen_dates: set[str] = set()
    for i, line in enumerate(log.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            problems.append(f"line {i}: not JSON")
            continue
        td, as_of = rec.get("trade_date"), rec.get("as_of")
        if not as_of:
            problems.append(f"line {i}: missing as_of")
            continue
        try:
            cur = _parse_as_of(as_of)
        except ValueError:
            problems.append(f"line {i}: unparseable as_of {as_of!r}")
            continue
        if prev_as_of is not None and cur < prev_as_of:
            problems.append(f"line {i}: as_of {as_of} < previous {prev_as_of.isoformat()} (rewind)")
        prev_as_of = cur
        if td:
            if str(td) > str(as_of)[:10]:
                problems.append(f"line {i}: trade_date {td} > as_of date {str(as_of)[:10]} (lookahead)")
            if td in seen_dates:
                problems.append(f"line {i}: duplicate trade_date {td}")
            seen_dates.add(td)
    return problems


def _verify_feed(snapshot: Path, as_of_date: str) -> tuple[str, list[str]]:
    """Reuse llm/scripts/verify_dividend_feed (independent no-lookahead). Returns (status, lines).

    status in {"PASS","FAIL","skipped"}. FAIL -> caller raises; skipped (missing inputs/deps) -> warn.
    """
    feed_csv = snapshot / "feed" / "dividend_calendar_upcoming.csv"
    edisc = snapshot / "feed" / "edisclosure"
    if not feed_csv.exists():
        return "skipped", ["no dividend feed in snapshot"]
    if not edisc.exists() or not any(edisc.glob("*.parquet")):
        return "skipped", ["no raw disclosure parquet bundled — cannot re-verify independently"]
    try:
        import pandas as pd  # noqa: F401
        sys.path.insert(0, str(REPO_ROOT / "llm" / "scripts"))
        import verify_dividend_feed as vdf  # type: ignore
    except Exception as exc:  # noqa: BLE001 - missing deps != a discrepancy
        return "skipped", [f"verifier unavailable ({type(exc).__name__}: {exc})"]

    import pandas as pd
    vdf.DDIR = edisc                      # point the verifier at the bundled raw titles
    try:
        feed = pd.read_csv(feed_csv)
        ok, lines, _ = vdf.verify(feed, pd.Timestamp(as_of_date))
    except Exception as exc:  # noqa: BLE001 - a torn/unreadable bundle is a FAIL, not a crash
        return "FAIL", [f"feed re-verify raised ({type(exc).__name__}: {exc}) — treat as not-verified"]
    return ("PASS" if ok else "FAIL"), lines


def _make_readonly(root: Path) -> None:
    for p in root.rglob("*"):
        if p.is_file():
            os.chmod(p, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)


def _make_writable(root: Path) -> None:
    for p in root.rglob("*"):
        try:
            os.chmod(p, stat.S_IWRITE | stat.S_IREAD)
        except OSError:
            pass


def import_snapshot(snapshot: Path, *, force: bool) -> dict:
    manifest_path = snapshot / "manifest.json"
    if not manifest_path.exists():
        raise VerifyError(f"no manifest.json in {snapshot}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("snapshot_type") != "forward_accrual":
        raise VerifyError(f"unexpected snapshot_type {manifest.get('snapshot_type')!r}")

    problems = _check_sha256(snapshot, manifest)
    problems += _check_shadow_track(snapshot)
    if problems:
        raise VerifyError("verification failed:\n  - " + "\n  - ".join(problems))

    as_of_date = (manifest.get("created_at") or "")[:10] or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    feed_status, feed_lines = _verify_feed(snapshot, as_of_date)
    if feed_status == "FAIL":
        raise VerifyError("dividend-feed no-lookahead FAILED:\n  " + "\n  ".join(feed_lines))

    # landing dir = the forward range end (max as_of) or the export date
    rng = manifest.get("as_of_range", {})
    landing_date = (rng.get("max") or "")[:10] or as_of_date
    target = SNAPSHOTS_ROOT / landing_date

    SNAPSHOTS_ROOT.mkdir(parents=True, exist_ok=True)
    (SNAPSHOTS_ROOT / ".gitignore").write_text("*\n!.gitignore\n", encoding="utf-8")
    if target.exists():
        if not force:
            raise VerifyError(f"{target} already imported (use --force to replace)")
        _make_writable(target)
        shutil.rmtree(target)
    shutil.copytree(snapshot, target)
    _make_readonly(target)

    return {
        "status": "ok",
        "landed": str(target),
        "read_only": True,
        "sha256_checked": len(manifest.get("files", [])),
        "shadow_track": "no-lookahead + monotonic OK",
        "feed_nolookahead": feed_status,
        "feed_notes": feed_lines if feed_status != "PASS" else [],
        "as_of_range": rng,
        "git_commit": manifest.get("git_commit"),
        "not_for_tuning": manifest.get("not_for_tuning"),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("snapshot", help="path to the exported snapshot dir (with manifest.json)")
    ap.add_argument("--force", action="store_true", help="replace an already-imported <date> dir")
    args = ap.parse_args(argv)

    try:
        result = import_snapshot(Path(args.snapshot), force=args.force)
    except VerifyError as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
