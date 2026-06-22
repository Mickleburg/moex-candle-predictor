"""Admin panel — the /users /allow /deny management commands (admin-only).

Pure of Telegram types: each method returns a reply string. The admin-only GATE lives in the
Router (config.is_admin) — these methods assume the caller is already an admin. They mutate ONLY
the bot-owned managed allowlist (config.allowlist); the bootstrap admins and the env seed are
environment-defined and intentionally NOT removable here (fail-safe against self-lockout).
"""

from __future__ import annotations

from . import formatters as fmt
from .config import BotConfig


class AdminPanel:
    def __init__(self, config: BotConfig):
        self.config = config
        self.store = config.allowlist

    def users(self) -> str:
        return fmt.fmt_users(self.config.admin_chat_ids, self.config.allowed_chat_ids,
                             self.store.entries())

    def allow(self, args: list[str], *, added_by: int | None = None) -> str:
        if not args:
            return "usage: /allow &lt;chat_id&gt; [note]"
        try:
            cid = int(args[0])
        except (ValueError, TypeError):
            return f"⛔ not an integer chat_id: {args[0]!r}"
        note = " ".join(args[1:]).strip()
        if cid in self.config.admin_chat_ids:
            return f"ℹ️ {cid} is an admin (👑) — already permanently allowed"
        if cid in self.config.allowed_chat_ids:
            return f"ℹ️ {cid} is in the env seed — already allowed"
        added = self.store.add(cid, note=note, added_by=added_by)
        if not added:
            return f"ℹ️ {cid} is already in the managed allowlist"
        return f"✅ allowed {cid}" + (f" ({note})" if note else "")

    def deny(self, args: list[str]) -> str:
        if not args:
            return "usage: /deny &lt;chat_id&gt;"
        try:
            cid = int(args[0])
        except (ValueError, TypeError):
            return f"⛔ not an integer chat_id: {args[0]!r}"
        if cid in self.config.admin_chat_ids:
            return f"⛔ {cid} is an admin (👑) — cannot be removed via the bot"
        if cid in self.config.allowed_chat_ids:
            return (f"⛔ {cid} is an env-seed id — remove it from BOT_ALLOWED_CHAT_IDS "
                    "(restart), not via the bot")
        removed = self.store.remove(cid)
        return f"✅ removed {cid}" if removed else f"ℹ️ {cid} was not in the managed allowlist"
