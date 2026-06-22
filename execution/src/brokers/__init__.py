"""Broker backends + the gated factory that maps a mode -> a concrete adapter."""

from __future__ import annotations

from ..config import LIVE_ENV_FLAG, ExecutionConfig, Mode
from .base import BrokerAdapter, execution_report
from .paper import DryRunBroker, PaperBroker
from .tinvest import TInvestBroker

__all__ = [
    "BrokerAdapter",
    "DryRunBroker",
    "PaperBroker",
    "TInvestBroker",
    "execution_report",
    "make_broker",
]


def _default_figi_map() -> dict[str, str]:
    from ..instruments import load_figi_map
    return load_figi_map()


def make_broker(config: ExecutionConfig, **broker_kwargs) -> BrokerAdapter:
    """Construct the adapter for ``config.mode``, enforcing the live gate.

    DRY_RUN -> DryRunBroker (sends nothing).
    PAPER   -> PaperBroker (internal sim, default) or TInvest sandbox if broker_backend=="tinvest".
    LIVE    -> TInvest production, but ONLY if config.live_enabled() (allow_live AND env flag) AND
               every backend FIGI is verified. Any gate unmet raises PermissionError — there is no
               accidental live path. The T-Invest paths auto-load the FIGI map from backend metadata.
    """
    if config.mode is Mode.DRY_RUN:
        return DryRunBroker()

    if config.mode is Mode.PAPER:
        if config.broker_backend == "tinvest":
            broker_kwargs.setdefault("figi_by_ticker", _default_figi_map())
            return TInvestBroker(sandbox=True, **broker_kwargs)
        return PaperBroker()

    if config.mode is Mode.LIVE:
        if not config.live_enabled():
            raise PermissionError(
                "LIVE trading is disabled. Require BOTH config.allow_live=True AND environment "
                f"{LIVE_ENV_FLAG}=1. Refusing to place real orders.")
        if config.broker_backend != "tinvest":
            raise PermissionError("LIVE requires broker_backend='tinvest' (a real broker).")
        from ..instruments import figis_all_verified
        if not figis_all_verified():
            raise PermissionError(
                "LIVE refused: backend reports unverified FIGIs. Verify them against a T-Invest "
                "dump (backend.instruments.all_verified()) before trading real money.")
        broker_kwargs.setdefault("figi_by_ticker", _default_figi_map())
        # Live limit prices are sourced from the broker's real-time quote (no paid data sub).
        broker_kwargs.setdefault("price_from_quote", True)
        return TInvestBroker(sandbox=False, **broker_kwargs)

    raise ValueError(f"unknown mode: {config.mode}")
