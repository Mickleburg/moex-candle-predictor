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

import asyncio
import logging

from . import heartbeat
from .config import BotConfig, load_bot_config
from .router import ALL_COMMANDS, Router

log = logging.getLogger("bot")

# Autocomplete menu (Telegram "/" popup). Plain (name, description) tuples so the data stays
# library-agnostic; converted to telegram.BotCommand only inside apply_command_menu. READ shows
# to everyone (default scope); READ+ADMIN is scoped to each admin chat so /users /allow /deny
# only surface for admins. Names mirror the real commands — never rename (BotFather + tests).
READ_MENU = [
    ("status", "Mode, kill-switch, last cycle, gross"),
    ("positions", "Live + shadow book"),
    ("pnl", "P&L by sleeve (live vs shadow)"),
    ("prices", "Last close [TICKERS]"),
    ("cycle", "Last EOD result"),
    ("integrity", "Data freshness gate"),
    ("gate", "H9 shadow gate"),
    ("shadowlog", "Forward-shadow track [N]"),
    ("start", "About this bot"),
    ("help", "Command list"),
]
ADMIN_MENU = [
    ("users", "Show the allowlist"),
    ("allow", "Grant read access: <id> [note]"),
    ("deny", "Revoke a managed id: <id>"),
]


async def apply_command_menu(bot, config: BotConfig) -> None:
    """Publish the "/" autocomplete menu (best-effort). Read commands go to the default scope;
    admin commands are scoped per admin chat. Any failure (network/proxy) is logged and swallowed
    so a menu hiccup never stops the bot from starting — same fail-safe stance as the notifier."""
    from telegram import BotCommand, BotCommandScopeChat, BotCommandScopeDefault

    read = [BotCommand(n, d) for n, d in READ_MENU]
    admin = [BotCommand(n, d) for n, d in (*READ_MENU, *ADMIN_MENU)]
    try:
        await bot.set_my_commands(read, scope=BotCommandScopeDefault())
    except Exception:  # noqa: BLE001 - menu is cosmetic; never block startup
        log.exception("set_my_commands (default scope) failed — non-fatal")
    for admin_id in sorted(config.admin_chat_ids):
        try:
            await bot.set_my_commands(admin, scope=BotCommandScopeChat(admin_id))
        except Exception:  # noqa: BLE001
            log.exception("set_my_commands (admin %s) failed — non-fatal", admin_id)


def build_get_updates_request(config: BotConfig):
    """The transport PTB uses for ``getUpdates`` ONLY — heartbeat + bounded timeouts.

    ``Bot._post`` routes the getUpdates endpoint to ``_request[0]`` and everything else (sendMessage,
    set_my_commands, get_me) to ``_request[1]``, so a stamp added here means "a poll round-trip
    completed", never "some unrelated API call worked". That distinction is the whole point: during
    both hangs the agent's notifier kept sending fine while the poller was dead.

    Two independent defences, because either alone leaves a gap:
      * heartbeat — makes the hang VISIBLE to the container healthcheck, which recycles the bot.
      * timeouts  — makes the hang IMPOSSIBLE to sustain: read_timeout bounds the await, so a
        half-open socket raises TimedOut, PTB retries, and the logs resume by themselves.

    NB on the read_timeout arithmetic: ``Bot.get_updates`` ADDS the long-poll timeout to the request
    object's read_timeout (``arg_read_timeout + timeout``), so the effective ceiling here is
    ``2*poll_timeout + 5`` = 65s at the default 30s. Bounded and comfortably under the 180s
    healthcheck threshold — a single slow poll can never trip a restart.

    The proxy is passed to the constructor rather than via ``builder.get_updates_proxy()``: PTB's
    ``_request_check`` makes a custom get_updates_request and the per-parameter setters mutually
    exclusive, so wiring one silently drops the other. Losing the proxy on a RU VDS = a bot that
    cannot reach Telegram at all.
    """
    from telegram.request import HTTPXRequest

    hb_path = config.heartbeat_path

    class _HeartbeatRequest(HTTPXRequest):
        async def do_request(self, *args, **kwargs):
            out = await super().do_request(*args, **kwargs)
            heartbeat.touch(hb_path)  # best-effort; swallows OSError so it cannot kill the poller
            return out

    return _HeartbeatRequest(
        proxy=config.proxy_url or None,
        connection_pool_size=1,        # matches PTB's own default for the getUpdates transport
        read_timeout=config.poll_timeout + 5,
        connect_timeout=10,
        pool_timeout=10,
    )


async def _run_watchdog(config: BotConfig, interval: float = heartbeat.WATCHDOG_INTERVAL) -> None:
    """In-process backstop that turns a wedged poller into an actual container restart.

    Runs as an asyncio task on the SAME loop as the poller. That is deliberate, not a compromise:
    the hang we hit twice was a getUpdates await on a half-open socket, and an await leaves the
    event loop itself alive — so this task keeps ticking and can act while the poller is frozen.
    (A fully-wedged loop, where even this task can't wake, is out of scope here by design: the
    compose healthcheck still flips the container to unhealthy for external visibility.)

    Ordering of defences: the bounded read_timeout on the getUpdates transport should raise TimedOut
    and let PTB self-heal LONG before the heartbeat goes stale, so in normal operation this loop
    only ever sees a fresh stamp and does nothing. It is the guaranteed backstop for the case the
    await does NOT raise — which is exactly what bit us on 07-15 and 07-20.
    """
    while True:
        await asyncio.sleep(interval)
        try:
            heartbeat.check_and_exit(config.heartbeat_path)
        except Exception:  # noqa: BLE001 - the backstop must never die silently on a stray error
            log.exception("watchdog check failed — will retry next tick")


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
    watchdog: list[asyncio.Task] = []  # holds the backstop task so post_shutdown can cancel it

    async def _post_init(application) -> None:
        # Stamp BEFORE the first poll completes: the healthcheck's start_period is short, and an
        # unstamped heartbeat during a slow start would read as "wedged" and flap the container.
        # The pre-stamp also gives the watchdog a real timestamp to measure from on its first tick.
        heartbeat.touch(config.heartbeat_path)
        # Raw create_task (not application.create_task): the watchdog is an infinite loop, and
        # application.create_task AWAITS its tasks on stop() — which would hang graceful shutdown.
        watchdog.append(asyncio.create_task(_run_watchdog(config), name="heartbeat-watchdog"))
        await apply_command_menu(application.bot, config)

    async def _post_shutdown(application) -> None:
        # Clean teardown on SIGTERM/SIGINT (irrelevant on the os._exit path — that kills outright).
        for task in watchdog:
            task.cancel()
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

    builder = (Application.builder().token(config.token)
               .post_init(_post_init).post_shutdown(_post_shutdown)
               # getUpdates gets its own instrumented transport (heartbeat + bounded timeouts);
               # it carries the proxy itself, hence no .get_updates_proxy() below.
               .get_updates_request(build_get_updates_request(config)))
    if config.proxy_url:
        # RU VDS can't reach api.telegram.org directly (RKN). This covers the bot's ordinary HTTP
        # client (sendMessage etc.); httpx handles an http:// proxy natively, no extra dep.
        builder = builder.proxy(config.proxy_url)
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
    log.info("bot starting — %d admin(s), %d seed id(s), polling getUpdates (timeout=%ds), "
             "heartbeat=%s", len(config.admin_chat_ids), len(config.allowed_chat_ids),
             config.poll_timeout, config.heartbeat_path)
    app.run_polling(timeout=config.poll_timeout, allowed_updates=["message"])
