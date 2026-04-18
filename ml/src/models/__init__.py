"""Model implementations."""

from .baseline import LogisticRegressionBaseline, MajorityClassifier, MarkovClassifier
from .lgbm_model import LGBMClassifier
from .rnn_model import RNNClassifier
from .train import main, train_pipeline

__all__ = [
    # Baseline models
    "MajorityClassifier",
    "MarkovClassifier",
    "LogisticRegressionBaseline",
    # Main MVP model
    "LGBMClassifier",
    # RNN stub (future implementation)
    "RNNClassifier",
    # Training pipeline
    "train_pipeline",
    "main",
]
