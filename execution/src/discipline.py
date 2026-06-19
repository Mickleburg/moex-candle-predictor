"""H9 entry/exit discipline guard.

The sleeve (ml) already decides WHICH names are in their pre-ex window; the risk_manager nets and
sizes them. This guard is execution's independent safety check that the book it is about to trade
respects the H9 timing law, given the dividend anchor (record/ex date) per name:

    a name should be HELD (target weight > 0) only when  exit_offset < td <= entry_offset,

where ``td`` = trading days from as_of to the anchor (counted on the trading calendar). The two
violations that matter:
  * held too late (td <= exit_offset): we are still long INTO the ex-gap — the exact loss the
    -2 exit exists to avoid. Flagged CRITICAL.
  * entered too early (td > entry_offset): premature exposure, no edge yet. Flagged WARN.

Anchors are optional: when the orchestrator does not pass an anchor for a name, that name is simply
not checked (reconciliation still runs). The guard never places or blocks orders by itself — the
engine decides what to do with the findings (log, or halt on a critical one).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

from .config import ExecutionConfig
from .trading_calendar import TradingCalendar, _as_date


@dataclass
class DisciplineFinding:
    instrument: str
    severity: str          # "ok" | "warn" | "critical"
    td_to_anchor: int      # trading days from as_of to the anchor
    held: bool             # is the target a non-zero long?
    in_window: bool        # does td fall in (exit, entry]?
    message: str


class DisciplineChecker:
    """Verify a target book against per-name dividend anchors and the H9 -entry/-exit window."""

    def __init__(self, config: ExecutionConfig | None = None,
                 calendar: TradingCalendar | None = None) -> None:
        self.config = config or ExecutionConfig()
        self.calendar = calendar or TradingCalendar()

    def _td(self, as_of: date, anchor: date) -> int:
        return self.calendar.trading_days_between(as_of, anchor)

    def check_name(self, instrument: str, target_weight: float, as_of, anchor) -> DisciplineFinding:
        as_of_d, anchor_d = _as_date(as_of), _as_date(anchor)
        td = self._td(as_of_d, anchor_d)
        held = target_weight > 0
        in_window = self.config.exit_offset < td <= self.config.entry_offset
        if held and td <= self.config.exit_offset:
            return DisciplineFinding(instrument, "critical", td, held, in_window,
                                     f"held with td={td} <= exit_offset={self.config.exit_offset}: "
                                     "long into the ex-gap — should have exited")
        if held and td > self.config.entry_offset:
            return DisciplineFinding(instrument, "warn", td, held, in_window,
                                     f"entered early: td={td} > entry_offset={self.config.entry_offset}")
        if held and not in_window:
            return DisciplineFinding(instrument, "warn", td, held, in_window,
                                     f"held outside the [{self.config.exit_offset+1}, "
                                     f"{self.config.entry_offset}] td window (td={td})")
        if not held and in_window:
            return DisciplineFinding(instrument, "warn", td, held, in_window,
                                     f"flat although td={td} is inside the entry window")
        return DisciplineFinding(instrument, "ok", td, held, in_window, "ok")

    def check_book(self, risk_book: dict, anchors: dict[str, object] | None) -> list[DisciplineFinding]:
        """Findings for every NAME (not hedge) that has an anchor in ``anchors``.

        ``anchors`` maps ticker -> record/ex date (date | datetime | ISO string). Names without an
        anchor are skipped. as_of is taken from the book.
        """
        if not anchors:
            return []
        as_of = risk_book.get("as_of")
        findings: list[DisciplineFinding] = []
        for p in risk_book.get("net_positions", []):
            tkr = p["ticker"]
            if tkr not in anchors:
                continue
            findings.append(self.check_name(tkr, float(p["weight"]), as_of, anchors[tkr]))
        return findings

    @staticmethod
    def has_critical(findings: list[DisciplineFinding]) -> bool:
        return any(f.severity == "critical" for f in findings)
