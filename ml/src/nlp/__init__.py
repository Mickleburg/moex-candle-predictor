"""NLP-style candle language pipeline.

This package implements the paper-inspired flow:
normalized candle shape -> candle word -> sentence window -> vector -> action.
"""

from .candles import (
    ACTION_LABELS,
    SentenceSamples,
    candle_shape_matrix,
    label_distribution,
    make_action_labels,
    make_sentence_samples,
    split_ranges,
)
from .classifiers import ClassifierSpec, build_classifier
from .clustering import CandleClusterer, ClusterSpec
from .pipeline import ExperimentConfig, run_experiment
from .vectorizers import VectorizerSpec, build_vectorizer

__all__ = [
    "ACTION_LABELS",
    "SentenceSamples",
    "candle_shape_matrix",
    "label_distribution",
    "make_action_labels",
    "make_sentence_samples",
    "split_ranges",
    "ClassifierSpec",
    "build_classifier",
    "CandleClusterer",
    "ClusterSpec",
    "ExperimentConfig",
    "run_experiment",
    "VectorizerSpec",
    "build_vectorizer",
]
