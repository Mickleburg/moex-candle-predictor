"""Monitor — maps a command to a rendered reply string. Pure of Telegram types.

This is the testable core: given a BotConfig + ReadOnlyState, each method fetches the relevant
read-only slice and renders it via ``formatters``. The Telegram layer (app.py) only wires these
methods to handlers, so the bot's behaviour can be unit-tested against a seeded state.sqlite
with no network and no Telegram library.
"""

from __future__ import annotations

from . import formatters as fmt
from .config import BotConfig
from .datasource import (
    ReadOnlyState,
    last_close,
    read_gate,
    read_integrity,
    sector_of,
)


class Monitor:
    def __init__(self, config: BotConfig, state: ReadOnlyState):
        self.config = config
        self.state = state

    def status(self) -> str:
        s = self.state
        return fmt.fmt_status({
            "mode": self.config.mode,
            "block_mode": self.config.block_mode,
            "live_enabled": self.config.live_enabled,
            "kill_switch": s.kill_switch_engaged(),
            "last_cycle": s.last_cycle(),
            "live_gross": s.gross("live"),
            "shadow_gross": s.gross("shadow"),
        })

    def positions(self) -> str:
        def _annotate(rows: list[dict]) -> list[dict]:
            for p in rows:
                p["sector"] = sector_of(p["ticker"])
            return rows
        live = _annotate(self.state.positions("live"))
        shadow = _annotate(self.state.positions("shadow"))
        return fmt.fmt_positions(live, shadow, self.config.capital_rub)

    def pnl(self) -> str:
        return fmt.fmt_pnl(
            self.state.pnl_by_sleeve("live"),
            self.state.pnl_by_sleeve("shadow"),
        )

    def prices(self, tickers: list[str] | None = None) -> str:
        names = [t.upper() for t in tickers] if tickers else list(self.config.universe)
        out: list[tuple[str, float | None]] = [
            (t, last_close(t, self.config.timeframe, self.config.data_raw)) for t in names
        ]
        return fmt.fmt_prices(out, self.config.timeframe)

    def gate(self) -> str:
        return fmt.fmt_gate(
            read_gate(self.config.gate_report),
            self.state.pnl_by_sleeve("shadow"),
        )

    def cycle(self) -> str:
        return fmt.fmt_cycle(self.state.latest_cycle("eod"))

    def integrity(self) -> str:
        return fmt.fmt_integrity(read_integrity(self.config.integrity_report))

    def help(self) -> str:
        return fmt.HELP_TEXT
