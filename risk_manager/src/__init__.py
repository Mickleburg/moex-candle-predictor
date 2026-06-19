"""risk_manager — V3 sleeve combiner + portfolio risk layer.

The combiner nets weakly-correlated strategy sleeves into one book and applies the shared
risk layer: vol-targeting (H4) x regime gate (H5) x limits (name/sector/gross/correlation),
with a book-level hedge (sector-preferred for the H9 dividend run-up). See `combiner.py`.
"""

from .combiner import (  # noqa: F401
    CombinerConfig,
    RiskBook,
    combine,
    to_risk_decisions,
)
