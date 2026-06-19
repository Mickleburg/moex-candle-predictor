"""Shared test fixtures + helpers for the agent block.

Builds an Orchestrator wired to a temp state store, mock adapters (with fault injection), and
a capturing notifier — so every cycle test runs offline, deterministically, stdlib-only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.src.adapters import mock  # noqa: E402
from agent.src.adapters.registry import Adapters  # noqa: E402
from agent.src.config import AgentConfig  # noqa: E402
from agent.src.orchestrator import Orchestrator  # noqa: E402
from agent.src.state_store import StateStore  # noqa: E402


class CapturingNotifier:
    def __init__(self):
        self.messages: list[tuple[str, str]] = []

    def send(self, subject: str, body: str) -> bool:
        self.messages.append((subject, body))
        return True

    def subjects(self) -> list[str]:
        return [s for s, _ in self.messages]


def make_config(tmp_path: Path, **overrides) -> AgentConfig:
    cfg = AgentConfig(
        mode=overrides.get("mode", "paper"),
        enable_live=overrides.get("enable_live", False),
        block_mode="mock",
        capital_rub=overrides.get("capital_rub", 10_000_000.0),
        state_db=tmp_path / "state.sqlite",
        cycle_results_dir=tmp_path / "cycles",
        shadow_log=tmp_path / "shadow.jsonl",
        log_dir=tmp_path / "logs",
    )
    cfg.ensure_dirs()
    return cfg


def make_orch(tmp_path: Path, *, backend=None, sleeve=None, combiner=None, execution=None,
              notifier=None, **cfg_overrides):
    cfg = make_config(tmp_path, **cfg_overrides)
    store = StateStore(cfg.state_db)
    adapters = Adapters(
        backend=backend or mock.MockBackend(universe=cfg.universe),
        sleeve=sleeve or mock.MockSleeve(),
        combiner=combiner or mock.MockCombiner(hedge_mode="sector"),
        execution=execution or mock.PaperBrokerExecution(),
        modes={"backend": "mock", "sleeve": "mock", "combiner": "mock", "execution": "mock"},
    )
    note = notifier or CapturingNotifier()
    orch = Orchestrator(cfg, store=store, adapters=adapters, notifier=note)
    return orch, store, note


@pytest.fixture
def orch_factory(tmp_path):
    def _factory(**kwargs):
        return make_orch(tmp_path, **kwargs)
    return _factory
