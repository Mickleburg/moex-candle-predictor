"""Managed allowlist store — the bot's OWN tiny JSON file (data/bot/allowlist.json).

The agent's SQLite state store is opened READ-ONLY by the bot, so runtime allowlist changes can
NOT live there. This is a separate, bot-owned file mapping ``{str(chat_id): {note, added_by,
added_at}}``. It holds ONLY the ids added at runtime via /allow — the bootstrap admins and the
``BOT_ALLOWED_CHAT_IDS`` env seed live in the environment and are never written here.

Durability: writes are atomic (temp file in the same dir + ``os.replace``), so a crash mid-write
never truncates the list. A missing/corrupt file reads as an empty allowlist. The file is read
fresh on each access so /allow / /deny take effect immediately, with no bot restart.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path


class AllowlistStore:
    def __init__(self, path: Path | str):
        self.path = Path(path)

    def entries(self) -> dict[int, dict]:
        """{chat_id: {note, added_by, added_at}}. Missing/corrupt file => {} (fail-safe)."""
        if not self.path.exists():
            return {}
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError, OSError):
            # corrupt bytes (invalid UTF-8) raise UnicodeDecodeError — a ValueError, NOT
            # JSONDecodeError/OSError; treat ANY unreadable file as an empty allowlist (fail-safe).
            return {}
        out: dict[int, dict] = {}
        if isinstance(raw, dict):
            for key, value in raw.items():
                try:
                    out[int(key)] = dict(value) if isinstance(value, dict) else {}
                except (ValueError, TypeError):
                    continue
        return out

    def ids(self) -> set[int]:
        return set(self.entries().keys())

    def add(self, chat_id: int, *, note: str = "", added_by: int | None = None) -> bool:
        """Add a managed id. Returns True if newly added, False if it was already present."""
        entries = self.entries()
        cid = int(chat_id)
        if cid in entries:
            return False
        entries[cid] = {
            "note": note,
            "added_by": added_by,
            "added_at": datetime.now(timezone.utc).isoformat(),
        }
        self._write(entries)
        return True

    def remove(self, chat_id: int) -> bool:
        """Remove a managed id. Returns True if it was present and removed."""
        entries = self.entries()
        cid = int(chat_id)
        if cid not in entries:
            return False
        entries.pop(cid)
        self._write(entries)
        return True

    def _write(self, entries: dict[int, dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {str(k): entries[k] for k in sorted(entries)}
        fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), prefix=".allowlist-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2)
            os.replace(tmp, self.path)  # atomic on POSIX + Windows
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
