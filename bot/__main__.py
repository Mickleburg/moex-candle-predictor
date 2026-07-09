"""Run the monitoring bot: ``python -m bot`` (loads .env / agent config, then polls)."""

from __future__ import annotations

from bot.src.app import run

if __name__ == "__main__":
    run()
