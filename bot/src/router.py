"""Pure command router: authorization + dispatch, free of Telegram types.

Maps an incoming ``(command, chat_id, args)`` to a reply string. Keeping the whole access-control
matrix here (not in the async PTB handler) means it is unit-tested offline:

  * not authorized            -> short "no access, your id is X" reply (never silent) + log
  * admin command, not admin  -> admin-only refusal
  * authorized                -> dispatch to Monitor (read) or AdminPanel (manage)

app.py wraps a single Router in one PTB handler + a fallback handler for unknown/plain messages.
"""

from __future__ import annotations

import logging

from . import formatters as fmt
from .admin import AdminPanel
from .config import BotConfig
from .datasource import make_state
from .monitor import Monitor

log = logging.getLogger("bot")

# read commands -> Monitor; help/start render the help text.
READ_COMMANDS = frozenset({
    "status", "positions", "pnl", "prices", "gate", "shadowlog", "cycle", "integrity",
})
ADMIN_COMMANDS = frozenset({"users", "allow", "deny"})
ALL_COMMANDS = READ_COMMANDS | ADMIN_COMMANDS | {"help", "start"}


def _first_int(args: list[str] | None) -> int | None:
    """First arg as int (the N in /shadowlog N); None if absent/non-numeric."""
    if not args:
        return None
    try:
        return int(args[0])
    except (ValueError, TypeError):
        return None


class Router:
    def __init__(self, config: BotConfig, monitor: Monitor | None = None,
                 admin: AdminPanel | None = None):
        self.config = config
        self.monitor = monitor or Monitor(config, make_state(config))
        self.admin = admin or AdminPanel(config)

    def dispatch(self, command: str, chat_id: int | None, args: list[str] | None = None) -> str:
        """Authorize + run a command. Always returns a reply string (never silent)."""
        args = args or []
        command = (command or "").lstrip("/").lower()

        if not self.config.authorized(chat_id):
            log.warning("denied non-whitelisted chat_id=%s cmd=/%s", chat_id, command)
            return fmt.unauthorized_text(chat_id)

        if command in ADMIN_COMMANDS and not self.config.is_admin(chat_id):
            log.warning("denied non-admin chat_id=%s admin-cmd=/%s", chat_id, command)
            return fmt.admin_only_text()

        return self._run(command, chat_id, args)

    def _run(self, command: str, chat_id: int | None, args: list[str]) -> str:
        if command in ("help", "start"):
            return self.monitor.help(self.config.is_admin(chat_id))
        if command == "prices":
            return self.monitor.prices(args or None)
        if command == "shadowlog":
            return self.monitor.shadowlog(_first_int(args))
        if command in READ_COMMANDS:
            return getattr(self.monitor, command)()
        if command == "users":
            return self.admin.users()
        if command == "allow":
            return self.admin.allow(args, added_by=chat_id)
        if command == "deny":
            return self.admin.deny(args)
        # authorized user, unrecognised command / plain text
        return fmt.unknown_command_text()
