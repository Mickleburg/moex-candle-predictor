"""Model implementations."""

from .baseline import LogisticRegressionBaseline, MajorityClassifier, MarkovClassifier
from .lgbm_model import LGBMClassifier


def train_pipeline(*args, **kwargs):
    """Lazy import to keep inference independent from training-only imports."""
    from .train import train_pipeline as _train_pipeline

    return _train_pipeline(*args, **kwargs)


def main(*args, **kwargs):
    """Lazy import to keep inference independent from training-only imports."""
    from .train import main as _main

    return _main(*args, **kwargs)

__all__ = [
    # Baseline models
    "MajorityClassifier",
    "MarkovClassifier",
    "LogisticRegressionBaseline",
    # Main MVP model
    "LGBMClassifier",
    # Training pipeline
    "train_pipeline",
    "main",
]
