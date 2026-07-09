"""Bot configuration — secrets + whitelist from the environment, paths from the agent config.

The bot is a READ-ONLY consumer of the agent's state, so it does not own a config of its own:
it reuses ``agent.src.config.load_config`` for the canonical paths (state DB, shadow log, cycle
results dir) and the trading universe, and takes only its own secrets (bot token) and access
control (allowed chat ids) from the environment. Nothing secret is ever read from a file in git.

Access control — two tiers:
  * admin   — BOT_ADMIN_CHAT_IDS (env bootstrap; the "root of trust"). May run management
              commands (/users /allow /deny) and can NEVER be removed via the bot (fail-safe).
  * allowed — may run read commands. Effective set = admins ∪ env seed ∪ managed store.
The managed store is the bot's own data/bot/allowlist.json (see allowlist.py); it is consulted
DYNAMICALLY so /allow / /deny take effect immediately without a restart.

Env vars (see .env.example):
  TELEGRAM_BOT_TOKEN     bot token from @BotFather (SHARED with the agent notifier; one token).
  BOT_ADMIN_CHAT_IDS     comma-separated admin chat ids (bootstrap; immutable via the bot).
  BOT_ALLOWED_CHAT_IDS   comma-separated SEED of read-only chat ids (runtime ids go to the store).
  BOT_POLL_TIMEOUT       long-poll timeout seconds for getUpdates (default 30).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from .allowlist import AllowlistStore

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INTEGRITY_REPORT = REPO_ROOT / "data" / "reports" / "data_integrity_status.json"
# The realized-P&L shadow gate verdict (owned by the ML chat's h9_shadow_pnl). Text report.
DEFAULT_GATE_REPORT = REPO_ROOT / "data" / "reports" / "h9_shadow_pnl.txt"
# The bot's OWN managed allowlist (gitignored). NOT the agent DB — that is opened read-only.
DEFAULT_ALLOWLIST = REPO_ROOT / "data" / "bot" / "allowlist.json"


@dataclass
class BotConfig:
    token: str | None = None
    admin_chat_ids: frozenset[int] = frozenset()       # bootstrap admins (immutable via the bot)
    allowed_chat_ids: frozenset[int] = frozenset()     # env SEED of read-only ids
    poll_timeout: int = 30
    proxy_url: str | None = None                       # TELEGRAM_PROXY_URL — RU hosts block Telegram

    # paths/universe mirrored from the agent config (read-only consumption)
    state_db: Path = REPO_ROOT / "data" / "agent" / "state.sqlite"
    shadow_log: Path = REPO_ROOT / "data" / "agent" / "shadow_pnl.jsonl"
    cycle_results_dir: Path = REPO_ROOT / "data" / "agent" / "cycles"
    integrity_report: Path = DEFAULT_INTEGRITY_REPORT
    gate_report: Path = DEFAULT_GATE_REPORT
    data_raw: Path = REPO_ROOT / "data" / "raw"
    allowlist_path: Path = DEFAULT_ALLOWLIST

    universe: list[str] = field(default_factory=list)
    capital_rub: float = 10_000_000.0
    timeframe: str = "1D"
    mode: str = "paper"
    block_mode: str = "mock"
    live_enabled: bool = False

    # the bot-owned managed allowlist store (built from allowlist_path if not injected)
    allowlist: AllowlistStore | None = None

    def __post_init__(self) -> None:
        if self.allowlist is None:
            self.allowlist = AllowlistStore(self.allowlist_path)

    def managed_ids(self) -> set[int]:
        """Runtime-added ids from the managed store (read fresh each call -> dynamic)."""
        return self.allowlist.ids() if self.allowlist else set()

    def is_admin(self, chat_id: int | None) -> bool:
        """True iff this chat is a bootstrap admin (may run management commands)."""
        return chat_id is not None and int(chat_id) in self.admin_chat_ids

    def authorized(self, chat_id: int | None) -> bool:
        """True iff this chat may use the bot: admin ∪ env seed ∪ managed store (dynamic)."""
        if chat_id is None:
            return False
        cid = int(chat_id)
        return cid in self.admin_chat_ids or cid in self.allowed_chat_ids or cid in self.managed_ids()

    def has_any_access(self) -> bool:
        """Any chat at all able to use the bot? (startup refuses if not — fail-closed)."""
        return bool(self.admin_chat_ids or self.allowed_chat_ids or self.managed_ids())


def _parse_chat_ids(raw: str | None, *, var_name: str = "BOT_ALLOWED_CHAT_IDS") -> frozenset[int]:
    if not raw:
        return frozenset()
    ids: set[int] = set()
    for part in raw.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            ids.add(int(part))
        except ValueError:
            # a malformed id must not silently widen access — skip it loudly at startup.
            print(f"[bot.config] ignoring non-integer chat id {part!r} in {var_name}")
    return frozenset(ids)


def _maybe_load_dotenv(path: Path = REPO_ROOT / ".env") -> None:
    """Best-effort, stdlib-only .env loader for LOCAL runs — fills only vars not already set.

    On the VDS the process manager (systemd EnvironmentFile / docker env_file) loads the env,
    so this is a convenience for `python -m bot` during local testing. Never overrides a value
    already present in the environment; silently does nothing if the file is absent.
    """
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_bot_config(config_path: Path | str | None = None, *, load_dotenv: bool = True) -> BotConfig:
    """Build the bot config: agent paths/universe + bot secrets/whitelist from the environment."""
    if load_dotenv:
        _maybe_load_dotenv()

    # Reuse the agent's canonical paths + universe. Imported lazily so a unit test can build a
    # BotConfig directly without the agent package on the path.
    from agent.src.config import load_config as load_agent_config

    agent = load_agent_config(config_path)
    return BotConfig(
        token=os.getenv("TELEGRAM_BOT_TOKEN") or None,
        admin_chat_ids=_parse_chat_ids(os.getenv("BOT_ADMIN_CHAT_IDS"), var_name="BOT_ADMIN_CHAT_IDS"),
        allowed_chat_ids=_parse_chat_ids(os.getenv("BOT_ALLOWED_CHAT_IDS")),
        poll_timeout=int(os.getenv("BOT_POLL_TIMEOUT", "30")),
        proxy_url=os.getenv("TELEGRAM_PROXY_URL") or None,
        allowlist_path=DEFAULT_ALLOWLIST,
        state_db=agent.state_db,
        shadow_log=agent.shadow_log,
        cycle_results_dir=agent.cycle_results_dir,
        data_raw=REPO_ROOT / "data" / "raw",
        universe=list(agent.universe),
        capital_rub=agent.capital_rub,
        timeframe=agent.timeframe,
        mode=agent.mode,
        block_mode=agent.block_mode,
        live_enabled=agent.live_enabled(),
    )
