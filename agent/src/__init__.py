"""Agent orchestrator source.

Light, dependency-free core (stdlib only): config, state-store, trading calendar,
contracts, notifier, P&L attribution, and the daily state-machine orchestrator. Heavy
dependencies (pandas / the ml + risk_manager packages) are imported LAZILY only inside
the `live` block adapters, so the core and the mock cycle run with stdlib alone.
"""
