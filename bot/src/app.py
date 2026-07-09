"""Telegram wiring — the ONLY module that imports python-telegram-bot.

Everything else (config, datasource, monitor, admin, router, formatters) is library-agnostic and
unit-tested offline; this module binds a single ``Router`` to PTB handlers and enforces nothing
itself — the Router owns the authorization matrix. ``python-telegram-bot`` is imported lazily
inside the functions so importing the bot package (and running its tests) never requires it.

Polling model: this bot is the single ``getUpdates`` consumer for the token (Application.run_polling).
The agent's notifier only ever calls ``sendMessage`` (push), so there is no getUpdates conflict.
Do NOT start a second poller on the same token.
"""

from __future__ import annotations

import logging

from .config import BotConfig, load_bot_config
from .router import ALL_COMMANDS, Router

log = logging.getLogger("bot")


def build_application(config: BotConfig):
    """Construct the PTB Application: one handler per command + a fallback for anything else."""
    from telegram import Update
    from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

    if not config.token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set — refusing to start the bot")
    if not config.has_any_access():
        # Fail-closed: with no admins AND no allowed ids the bot answers nobody. Stop at startup
        # rather than look "up" while silently muted.
        raise RuntimeError("no admins or allowed chat ids — set BOT_ADMIN_CHAT_IDS / "
                           "BOT_ALLOWED_CHAT_IDS before start")

    router = Router(config)
    notified_unauth: set[int] = set()  # one admin notification per unknown id per process

    def _make_handler(command: str | None):
        async def _handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            chat = update.effective_chat
            chat_id = chat.id if chat else None
            # CommandHandlers pass their name; the fallback passes None -> empty -> "unknown".
            cmd = command if command is not None else ""
            try:
                text = router.dispatch(cmd, chat_id, list(context.args) if context.args else [])
            except Exception:  # noqa: BLE001 - a read/format error must not kill the poller
                log.exception("command /%s failed", cmd)
                text = "⚠️ internal error reading agent state — see bot logs"

            if chat_id is not None and not config.authorized(chat_id) \
                    and chat_id not in notified_unauth:
                notified_unauth.add(chat_id)
                await _notify_admins(context, config, chat_id)

            if update.effective_message:
                await update.effective_message.reply_text(text, parse_mode="HTML")
        return _handler

    builder = Application.builder().token(config.token)
    if config.proxy_url:
        # RU VDS can't reach api.telegram.org directly (RKN). Route BOTH the bot's HTTP client and
        # the getUpdates poller through the proxy (httpx handles an http:// proxy natively, no dep).
        builder = builder.proxy(config.proxy_url).get_updates_proxy(config.proxy_url)
    app = builder.build()
    for command in sorted(ALL_COMMANDS):
        app.add_handler(CommandHandler(command, _make_handler(command)))
    # fallback: unknown commands / plain text -> unauthorized notice or a /help hint (added last
    # so registered command handlers win first within the default group).
    app.add_handler(MessageHandler(filters.ALL, _make_handler(None)))
    return app


async def _notify_admins(context, config: BotConfig, chat_id: int) -> None:
    """Best-effort: tell admins someone requested access (self-service for 'what's my id')."""
    msg = f"🔔 access request — chat_id {chat_id} tried to use the bot. Reply /allow {chat_id} to grant."
    for admin_id in config.admin_chat_ids:
        try:
            await context.bot.send_message(admin_id, msg)
        except Exception:  # noqa: BLE001 - notification is best-effort
            log.exception("failed to notify admin %s", admin_id)


def run(config: BotConfig | None = None) -> None:
    """Start the long-lived poller (blocking). Single getUpdates consumer for the token."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    config = config or load_bot_config()
    app = build_application(config)
    log.info("bot starting — %d admin(s), %d seed id(s), polling getUpdates (timeout=%ds)",
             len(config.admin_chat_ids), len(config.allowed_chat_ids), config.poll_timeout)
    app.run_polling(timeout=config.poll_timeout, allowed_updates=["message"])
