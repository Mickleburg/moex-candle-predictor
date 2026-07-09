"""Pytest fixtures for the execution block.

Adds the repo root to sys.path (so `import execution.src...` works) and provides an ExecutionConfig
whose audit/state dirs are redirected into a tmp path — tests never touch execution/var.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from execution.src.config import ExecutionConfig, Mode  # noqa: E402


@pytest.fixture
def tmp_config(tmp_path: Path):
    """Factory: build an ExecutionConfig with isolated audit/state dirs under tmp_path."""

    def _make(mode: Mode = Mode.PAPER, **kwargs) -> ExecutionConfig:
        return ExecutionConfig(
            mode=mode,
            audit_dir=tmp_path / "audit",
            state_dir=tmp_path / "state",
            **kwargs,
        )

    return _make
