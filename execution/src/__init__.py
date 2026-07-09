"""Execution block source.

Public surface:

    from execution.src import (
        ExecutionConfig, Mode, SanityLimits,
        TradingCalendar,
        reconcile, DeltaOrder,
        DisciplineChecker, DisciplineFinding,
        AuditLog,
        make_broker, BrokerAdapter, DryRunBroker, PaperBroker,
        ExecutionEngine, CycleResult,
    )
"""

from __future__ import annotations

from .audit import AuditLog
from .brokers import BrokerAdapter, DryRunBroker, PaperBroker, make_broker
from .config import DEFAULT_LOT_SIZES, ExecutionConfig, Mode, SanityLimits
from .discipline import DisciplineChecker, DisciplineFinding
from .engine import CycleResult, ExecutionEngine
from .instruments import load_figi_map, load_lot_sizes
from .reconcile import DeltaOrder, reconcile
from .trading_calendar import TradingCalendar, active_calendar_source, default_trading_calendar

__all__ = [
    "AuditLog",
    "BrokerAdapter",
    "CycleResult",
    "DEFAULT_LOT_SIZES",
    "DeltaOrder",
    "DisciplineChecker",
    "DisciplineFinding",
    "DryRunBroker",
    "ExecutionConfig",
    "ExecutionEngine",
    "Mode",
    "PaperBroker",
    "SanityLimits",
    "TradingCalendar",
    "active_calendar_source",
    "default_trading_calendar",
    "load_figi_map",
    "load_lot_sizes",
    "make_broker",
    "reconcile",
]
