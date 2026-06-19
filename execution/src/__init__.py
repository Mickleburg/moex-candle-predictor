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
from .trading_calendar import TradingCalendar
from .config import DEFAULT_LOT_SIZES, ExecutionConfig, Mode, SanityLimits
from .discipline import DisciplineChecker, DisciplineFinding
from .engine import CycleResult, ExecutionEngine
from .reconcile import DeltaOrder, reconcile

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
    "make_broker",
    "reconcile",
]
