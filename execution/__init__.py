"""Execution block — broker adapter (dry-run/paper/live) + order discipline for the V3 agent.

Step 6 of the VDS daily cycle (docs/VDS_AUTONOMOUS_PLAN.md): take the target book from the
risk_manager (`risk_book`), reconcile it against current positions, emit lot-rounded LIMIT delta
orders under the H9 dividend run-up discipline (enter ~-12 trading days / exit ~-2 before the
ex-gap), and keep a full audit. Live trading is OFF by default and gated behind an explicit flag.
"""
