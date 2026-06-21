"""Telegram wiring — the ONLY module that imports python-telegram-bot.

Everything else (config, datasource, monitor, formatters) is library-agnostic and unit-tested
offline; this module just binds the Monitor's methods to commands and enforces the chat-id
whitelist. ``python-telegram-bot`` is imported lazily inside the functions so importing the bot
package (and running its tests) never requires the library to be installed.

Polling model: this bot is the single ``getUpdates`` consumer for the token (Application.run_polling).
The agent's notifier only ever calls ``sendMessage`` (push), so there is no getUpdates conflict.
Do NOT start a second poller on the same token.
"""

from __future__ import annotations

import logging

from .config import BotConfig, load_bot_config
from .datasource import make_state
from .monitor import Monitor

log = logging.getLogger("bot")

# command name -> Monitor method name. /prices additionally consumes context.args.
_COMMANDS = {
    "status": "status",
    "positions": "positions",
    "pnl": "pnl",
    "prices": "prices",
    "gate": "gate",
    "shadowlog": "shadowlog",
    "cycle": "cycle",
    "integrity": "integrity",
    "help": "help",
    "start": "help",
}


def _first_int(args: list[str] | None) -> int | None:
    """Parse the first CLI arg as a positive int (the N in /shadowlog N); None if absent/bad."""
    if not args:
        return None
    try:
        return int(args[0])
    except (ValueError, TypeError):
        return None


def build_application(config: BotConfig):
    """Construct the PTB Application with whitelisted command handlers. Requires a token."""
    from telegram import Update
    from telegram.ext import Application, CommandHandler, ContextTypes

    if not config.token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set — refusing to start the bot")
    if not config.allowed_chat_ids:
        # Fail loud: a bot with an empty whitelist answers nobody and is almost certainly a
        # misconfiguration. Better to stop at startup than to look "up" but silently mute.
        raise RuntimeError("BOT_ALLOWED_CHAT_IDS is empty — set the owner chat id(s) before start")

    monitor = Monitor(config, make_state(config))

    def _make_handler(method_name: str):
        async def _handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            chat = update.effective_chat
            chat_id = chat.id if chat else None
            if not config.authorized(chat_id):
                log.warning("ignoring update from non-whitelisted chat_id=%s", chat_id)
                return
            try:
                if method_name == "prices":
                    text = monitor.prices(context.args or None)
                elif method_name == "shadowlog":
                    text = monitor.shadowlog(_first_int(context.args))
                else:
                    text = getattr(monitor, method_name)()
            except Exception:  # noqa: BLE001 - a read/format error must not kill the poller
                log.exception("command /%s failed", method_name)
                text = "⚠️ internal error reading agent state — see bot logs"
            await update.effective_message.reply_text(text, parse_mode="HTML")
        return _handler

    app = Application.builder().token(config.token).build()
    for command, method_name in _COMMANDS.items():
        app.add_handler(CommandHandler(command, _make_handler(method_name)))
    return app


def run(config: BotConfig | None = None) -> None:
    """Start the long-lived poller (blocking). Single getUpdates consumer for the token."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    config = config or load_bot_config()
    app = build_application(config)
    log.info("bot starting — %d whitelisted chat(s), polling getUpdates (timeout=%ds)",
             len(config.allowed_chat_ids), config.poll_timeout)
    app.run_polling(timeout=config.poll_timeout, allowed_updates=["message"])
