"""Wire the block adapters from config.

Global `block_mode` (mock|live) sets the default; each block may override via
blocks.<name>.mode. This lets, e.g., the real ML sleeve + combiner run while execution stays
on the paper-broker mock — the common bring-up before the execution block lands.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import AgentConfig
from .base import BackendAdapter, CombinerAdapter, ExecutionAdapter, SleeveAdapter


@dataclass
class Adapters:
    backend: BackendAdapter
    sleeve: SleeveAdapter
    combiner: CombinerAdapter
    execution: ExecutionAdapter
    modes: dict[str, str]


def _mode_for(config: AgentConfig, block: str) -> str:
    return str(config.blocks.get(block, {}).get("mode", config.block_mode))


def build_adapters(config: AgentConfig) -> Adapters:
    from . import mock

    backend_cfg = config.blocks.get("backend", {})
    sleeve_cfg = config.blocks.get("sleeve", {})
    combiner_cfg = config.blocks.get("combiner", {})
    execution_cfg = config.blocks.get("execution", {})

    # --- backend ---
    if _mode_for(config, "backend") == "live":
        from .live import LiveBackend
        backend: BackendAdapter = LiveBackend(backend_cfg)
    else:
        backend = mock.MockBackend(universe=config.universe)

    # --- sleeve (ML) ---
    if _mode_for(config, "sleeve") == "live":
        from .live import LiveSleeve
        sleeve: SleeveAdapter = LiveSleeve(
            config.universe, timeframe=config.timeframe,
            model_version=sleeve_cfg.get("model_version", "h9_dividend_runup_v1"),
            command=sleeve_cfg.get("command"))
    else:
        sleeve = mock.MockSleeve(model_version=sleeve_cfg.get("model_version",
                                                              "h9_dividend_runup_v1_mock"))

    # --- combiner (risk_manager) ---
    if _mode_for(config, "combiner") == "live":
        from .live import LiveCombiner
        combiner: CombinerAdapter = LiveCombiner(
            config.universe, timeframe=config.timeframe,
            hedge_mode=combiner_cfg.get("hedge_mode", "sector"),
            target_book_vol_annual=combiner_cfg.get("target_book_vol_annual", 0.12))
    else:
        combiner = mock.MockCombiner(hedge_mode=combiner_cfg.get("hedge_mode", "sector"),
                                     timeframe=config.timeframe)

    # --- execution ---
    if _mode_for(config, "execution") == "live":
        from .live import LiveExecution
        execution: ExecutionAdapter = LiveExecution(execution_cfg)
    else:
        execution = mock.PaperBrokerExecution()

    return Adapters(
        backend=backend, sleeve=sleeve, combiner=combiner, execution=execution,
        modes={
            "backend": _mode_for(config, "backend"),
            "sleeve": _mode_for(config, "sleeve"),
            "combiner": _mode_for(config, "combiner"),
            "execution": _mode_for(config, "execution"),
        },
    )
