"""Single source of truth for the H9 dividend-sleeve universe + the forward bucket split.

Why this module exists: the H9 research toolchain (research / sim / cost / robustness / no-lookahead),
the realized-P&L gate, and the serving/monitor CLIs all need the SAME universe and the SAME forward
bucket rule. Duplicated lists drift; this centralizes them.

UNIVERSES
  CURRENT_UNIVERSE  — the 16 data-backed lines actually traded/validated today.
  EXPANDED_UNIVERSE — the a-priori-fixed 24-line list (see
    ml/docs/research/h9_universe_expansion_2026-06-21.md). DATA-GATED: the 8 extra lines need backend
    candles + ISS dividend history before they contribute. Until then they are simply skipped by the
    loaders (the IS study + gate load per-name and drop missing; load_daily_panel WARNs+skips), so
    selecting EXPANDED is HARMLESS before the data lands — it just reproduces CURRENT.

Selecting the universe for a run (one-command, no source edits):
  - env  H9_UNIVERSE=current|expanded   -> flips the whole research toolchain via active_universe().
  - or   --universe current|expanded    on h9_dividend_research.py / h9_shadow_pnl.py.
  SERVING (predict_dividend_sleeve, dividend_sleeve_monitor) deliberately PINS CURRENT_UNIVERSE: the
  serving universe is promoted only by a one-line edit AFTER the expanded IS edge + placebo control
  pass (lead's discipline; is_production stays false until the gate is MET on PRISTINE forward).

FORWARD BUCKET SPLIT (lead decision, fixed a-priori 2026-06-21 — does NOT lower the gate)
  Forward events (record >= FORWARD_START) split into two buckets; ONLY pristine drives is_production:
    PRISTINE       record OUTSIDE the burned-split backfill window
                   = [FORWARD_START, 2025-08-01)  OR  (2026-06-30, +inf)
                   = the existing 2025 forward (n=12) + the July-2026 wave + autumn 2026+.
    CORROBORATION  record in [2025-08-01, 2026-06-30] = the backfill of the burned directional split
                   (2025-09->2026-06). Reported SEPARATELY as robustness; sign agreement with pristine
                   = confirmation, divergence = red flag. Does NOT move the gate.
  Gate threshold is UNCHANGED: PRISTINE n >= 25 AND net > 0 AND %pos > 0.5.

Caveat for panel-based scripts: load_daily_panel intersects histories (dropna how=any), so when ragged
short-history lines land they may truncate the panel; the IS study + gate load per-name and are robust.
"""
from __future__ import annotations

import os

import pandas as pd

CURRENT_UNIVERSE = ["SBER", "GAZP", "LKOH", "GMKN", "ROSN", "NVTK", "TATN", "MGNT",
                    "MTSS", "SNGS", "CHMF", "ALRS", "VTBR", "MAGN", "NLMK", "PLZL"]

# a-priori expanded list (24 lines); the 8 additions are DATA-GATED + 2 (RTKMP, BSPB) provisional on
# backend's ADTV screen. Inert until backend delivers candles + ISS history.
EXPANDED_ADDITIONS = ["SBERP", "SNGSP", "TATNP", "SIBN", "PHOR", "RTKMP", "MOEX", "BSPB"]
EXPANDED_UNIVERSE = CURRENT_UNIVERSE + EXPANDED_ADDITIONS

UNIVERSES = {"current": CURRENT_UNIVERSE, "expanded": EXPANDED_UNIVERSE}
UNIVERSE_CHOICES = sorted(UNIVERSES)


def resolve_universe(name: str = "current") -> list[str]:
    if name not in UNIVERSES:
        raise ValueError(f"unknown universe '{name}'; choose from {UNIVERSE_CHOICES}")
    return list(UNIVERSES[name])


def active_universe() -> list[str]:
    """Universe for the research toolchain, env-selectable for one-command expanded runs.
    H9_UNIVERSE=expanded flips research/sim/cost/robustness at import; default 'current'."""
    return resolve_universe(os.environ.get("H9_UNIVERSE", "current"))


# --- forward bucket split (tz Moscow), fixed a-priori --------------------------------------------
FORWARD_START = pd.Timestamp("2025-01-01", tz="Europe/Moscow")
CORROBORATION_START = pd.Timestamp("2025-08-01", tz="Europe/Moscow")  # burned-split backfill begins
CORROBORATION_END = pd.Timestamp("2026-06-30", tz="Europe/Moscow")    # pristine-new resumes 2026-07-01


def classify_forward(record_date: pd.Timestamp) -> str:
    """'pristine' (drives the gate) | 'corroboration' (robustness only) for a forward record date."""
    if CORROBORATION_START <= record_date <= CORROBORATION_END:
        return "corroboration"
    return "pristine"
