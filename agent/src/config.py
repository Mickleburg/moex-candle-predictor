"""Agent configuration: a JSON config file overlaid with environment variables.

JSON (not YAML) on purpose — the orchestrator core stays stdlib-only, no PyYAML. Secrets
NEVER live in the config file; they come from the environment (.env on the VDS, loaded by
the process manager). See agent/config/agent_config.json and .env.example.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "agent" / "config" / "agent_config.json"

# H9 sleeve universe (16 names) — mirrors backend INSTRUMENT_REGISTRY / ml universe.
DEFAULT_UNIVERSE = [
    "SBER", "GAZP", "LKOH", "GMKN", "ROSN", "NVTK", "TATN", "MGNT",
    "MTSS", "SNGS", "CHMF", "ALRS", "VTBR", "MAGN", "NLMK", "PLZL",
]


@dataclass
class ScheduleConfig:
    timezone: str = "Europe/Moscow"
    eod: str = "19:05"          # after the main clearing
    preopen: str = "09:30"      # before the open
    deadman_check_minutes: int = 30     # how often the dead-man's-switch checks
    deadman_max_stale_hours: float = 26.0   # alert if no successful cycle within this window


@dataclass
class AlertConfig:
    channel: str = "stdout"     # "stdout" (dev/paper default) | "telegram"
    # telegram_bot_token / telegram_chat_id come from the environment, never the file.


@dataclass
class AgentConfig:
    # Trading mode for execution: live is gated behind enable_live AND mode == "live".
    mode: str = "paper"                 # "dry-run" | "paper" | "live"
    enable_live: bool = False           # hard gate: live is impossible unless this is true
    block_mode: str = "mock"            # "mock" (default; runs without other blocks) | "live"
    universe: list[str] = field(default_factory=lambda: list(DEFAULT_UNIVERSE))
    capital_rub: float = 10_000_000.0
    timeframe: str = "1D"

    state_db: Path = REPO_ROOT / "data" / "agent" / "state.sqlite"
    cycle_results_dir: Path = REPO_ROOT / "data" / "agent" / "cycles"
    shadow_log: Path = REPO_ROOT / "data" / "agent" / "shadow_pnl.jsonl"
    log_dir: Path = REPO_ROOT / "data" / "agent" / "logs"

    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)
    # Free-form per-block settings handed to the adapters (commands, hedge mode, etc.).
    blocks: dict[str, Any] = field(default_factory=dict)

    def ensure_dirs(self) -> None:
        for p in (self.state_db.parent, self.cycle_results_dir, self.shadow_log.parent, self.log_dir):
            p.mkdir(parents=True, exist_ok=True)

    def live_enabled(self) -> bool:
        """Live trading requires BOTH the explicit flag AND mode=live (paper-first invariant)."""
        return bool(self.enable_live) and self.mode == "live"


def _as_path(value: Any, default: Path) -> Path:
    if value is None:
        return default
    p = Path(str(value))
    return p if p.is_absolute() else (REPO_ROOT / p)


def load_config(path: Path | str | None = None) -> AgentConfig:
    """Load the agent config from JSON, then overlay environment variables.

    Missing file -> all defaults. Env overrides (handy on the VDS / in CI):
      AGENT_MODE, AGENT_ENABLE_LIVE, AGENT_BLOCK_MODE, AGENT_CAPITAL_RUB,
      AGENT_ALERT_CHANNEL.
    """
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    raw: dict[str, Any] = {}
    if cfg_path.exists():
        raw = json.loads(cfg_path.read_text(encoding="utf-8"))

    paths = raw.get("paths", {})
    sched = raw.get("schedule", {})
    alerts = raw.get("alerts", {})

    cfg = AgentConfig(
        mode=str(raw.get("mode", "paper")),
        enable_live=bool(raw.get("enable_live", False)),
        block_mode=str(raw.get("block_mode", "mock")),
        universe=list(raw.get("universe", DEFAULT_UNIVERSE)),
        capital_rub=float(raw.get("capital_rub", 10_000_000.0)),
        timeframe=str(raw.get("timeframe", "1D")),
        state_db=_as_path(paths.get("state_db"), AgentConfig.state_db),
        cycle_results_dir=_as_path(paths.get("cycle_results_dir"), AgentConfig.cycle_results_dir),
        shadow_log=_as_path(paths.get("shadow_log"), AgentConfig.shadow_log),
        log_dir=_as_path(paths.get("log_dir"), AgentConfig.log_dir),
        schedule=ScheduleConfig(
            timezone=str(sched.get("timezone", "Europe/Moscow")),
            eod=str(sched.get("eod", "19:05")),
            preopen=str(sched.get("preopen", "09:30")),
            deadman_check_minutes=int(sched.get("deadman_check_minutes", 30)),
            deadman_max_stale_hours=float(sched.get("deadman_max_stale_hours", 26.0)),
        ),
        alerts=AlertConfig(channel=str(alerts.get("channel", "stdout"))),
        blocks=dict(raw.get("blocks", {})),
    )

    # --- environment overlay (deploy / CI) ---
    if os.getenv("AGENT_MODE"):
        cfg.mode = os.environ["AGENT_MODE"]
    if os.getenv("AGENT_ENABLE_LIVE"):
        cfg.enable_live = os.environ["AGENT_ENABLE_LIVE"].lower() in {"1", "true", "yes"}
    if os.getenv("AGENT_BLOCK_MODE"):
        cfg.block_mode = os.environ["AGENT_BLOCK_MODE"]
    if os.getenv("AGENT_CAPITAL_RUB"):
        cfg.capital_rub = float(os.environ["AGENT_CAPITAL_RUB"])
    if os.getenv("AGENT_ALERT_CHANNEL"):
        cfg.alerts.channel = os.environ["AGENT_ALERT_CHANNEL"]

    return cfg
