"""Append-only audit log: every order, fill, rejection, cancel, kill, and discipline finding.

One JSON object per line (JSONL) under ``audit_dir`` (gitignored: it is runtime state, not source).
Append-only by construction — the engine never rewrites past lines — so the file is a faithful,
ordered record for the daily digest, the paper<->sim reconciliation, and post-mortems.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

MSK = timezone.utc  # records carry an explicit UTC stamp; wall-clock MSK is derivable downstream


@dataclass
class AuditLog:
    audit_dir: Path
    is_production: bool = False

    def __post_init__(self) -> None:
        self.audit_dir = Path(self.audit_dir)
        self.audit_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, day: str | None = None) -> Path:
        day = day or datetime.now(MSK).strftime("%Y%m%d")
        return self.audit_dir / f"audit-{day}.jsonl"

    def record(self, event: str, payload: dict, *, day: str | None = None) -> dict:
        """Append one event and return the written record (for tests / chaining)."""
        rec = {
            "ts": datetime.now(MSK).isoformat(),
            "event": event,
            "is_production": self.is_production,
            "payload": payload,
        }
        path = self._path(day)
        line = json.dumps(rec, ensure_ascii=False)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return rec

    def read_day(self, day: str) -> list[dict]:
        path = self._path(day)
        if not path.exists():
            return []
        return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
