"""Bot configuration — secrets + whitelist from the environment, paths from the agent config.

The bot is a READ-ONLY consumer of the agent's state, so it does not own a config of its own:
it reuses ``agent.src.config.load_config`` for the canonical paths (state DB, shadow log, cycle
results dir) and the trading universe, and takes only its own secrets (bot token) and access
control (allowed chat ids) from the environment. Nothing secret is ever read from a file in git.

Env vars (see .env.example):
  TELEGRAM_BOT_TOKEN     bot token from @BotFather (SHARED with the agent notifier; one token).
  BOT_ALLOWED_CHAT_IDS   comma-separated owner chat ids; the bot answers ONLY these. Empty => nobody.
  BOT_POLL_TIMEOUT       long-poll timeout seconds for getUpdates (default 30).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INTEGRITY_REPORT = REPO_ROOT / "data" / "reports" / "data_integrity_status.json"
# The realized-P&L shadow gate verdict (owned by the ML chat's h9_shadow_pnl). Text report.
DEFAULT_GATE_REPORT = REPO_ROOT / "data" / "reports" / "h9_shadow_pnl.txt"


@dataclass
class BotConfig:
    token: str | None = None
    allowed_chat_ids: frozenset[int] = frozenset()
    poll_timeout: int = 30

    # paths/universe mirrored from the agent config (read-only consumption)
    state_db: Path = REPO_ROOT / "data" / "agent" / "state.sqlite"
    shadow_log: Path = REPO_ROOT / "data" / "agent" / "shadow_pnl.jsonl"
    cycle_results_dir: Path = REPO_ROOT / "data" / "agent" / "cycles"
    integrity_report: Path = DEFAULT_INTEGRITY_REPORT
    gate_report: Path = DEFAULT_GATE_REPORT
    data_raw: Path = REPO_ROOT / "data" / "raw"

    universe: list[str] = field(default_factory=list)
    capital_rub: float = 10_000_000.0
    timeframe: str = "1D"
    mode: str = "paper"
    block_mode: str = "mock"
    live_enabled: bool = False

    def authorized(self, chat_id: int | None) -> bool:
        """True iff this chat is whitelisted. No whitelist => the bot answers nobody."""
        return chat_id is not None and int(chat_id) in self.allowed_chat_ids


def _parse_chat_ids(raw: str | None) -> frozenset[int]:
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
            print(f"[bot.config] ignoring non-integer chat id {part!r} in BOT_ALLOWED_CHAT_IDS")
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
        allowed_chat_ids=_parse_chat_ids(os.getenv("BOT_ALLOWED_CHAT_IDS")),
        poll_timeout=int(os.getenv("BOT_POLL_TIMEOUT", "30")),
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
