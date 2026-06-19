"""Backend / data block for the MOEX trading agent (V3, Python).

Owns the autonomous data foundation of the daily cycle (docs/VDS_AUTONOMOUS_PLAN.md):
    step 1 (EOD)  -- idempotent incremental ingest of fresh candles + market context
    step 3 (EOD)  -- data-integrity gate (freshness / gaps / NaN) -> OK | HALT

plus a shared MOEX trading calendar (RU-holiday aware) that ML / agent import so the
trading-day counters used for sleeve entry/exit timing stop drifting across holidays.

Storage is the existing file-based parquet store in ``data/raw`` (regenerable artifacts,
gitignored). No HTTP service: a single VDS reads the shared files directly -- see
docs/VDS_AUTONOMOUS_PLAN.md and backend/README.md for the rationale.
"""

from __future__ import annotations

__all__ = ["trading_calendar", "store", "ingest", "integrity", "universe"]
