"""MOEX trading agent — V3 orchestrator block.

The autonomous daily cycle around the H9 dividend run-up sleeve. See agent/README.md.
This package is the GLUE: it calls the other blocks (backend/data, ml, risk_manager,
execution) ONLY through their JSON contracts / public CLIs, never by editing their code.
"""
