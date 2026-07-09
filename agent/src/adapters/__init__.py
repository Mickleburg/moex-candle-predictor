"""Block adapters: the only place the agent touches another block.

Each block (backend/data, ml sleeve, risk_manager combiner, execution) sits behind an
interface with a `mock` and a `live` implementation. The mock path lets the full daily
cycle run end-to-end today, before backend/execution land; the live path calls the real
block's public API (ml, risk_manager — read-only) or CLI (backend, execution) and validates
the JSON it returns against contracts/. `build_adapters(config)` wires the set by mode.
"""

from .base import (  # noqa: F401
    BackendAdapter,
    CombinerAdapter,
    ExecutionAdapter,
    ExecutionResult,
    IntegrityStatus,
    SleeveAdapter,
)
from .registry import build_adapters  # noqa: F401
