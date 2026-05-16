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
from .action_features import LMActionFeatures, make_lm_action_features
from .classifiers import ClassifierSpec, build_classifier
from .clustering import CandleClusterer, ClusterSpec
from .pipeline import ExperimentConfig, run_experiment
from .vectorizers import VectorizerSpec, build_vectorizer
from .word_forecast import (
    NextWordSamples,
    build_word_forecaster,
    clusterer_distance_matrix,
    evaluate_word_forecast,
    expected_next_word_sample_count,
    fit_markov_prior_features,
    make_markov_prior_feature_matrix,
    make_next_word_samples,
)
from .word_lm import (
    NGramBackoffLanguageModel,
    confidence_analysis,
    error_analysis,
    evaluate_language_model,
    transition_entropy,
    transition_quality_metrics,
    word_distribution_metrics,
)

__all__ = [
    "ACTION_LABELS",
    "LMActionFeatures",
    "SentenceSamples",
    "candle_shape_matrix",
    "label_distribution",
    "make_action_labels",
    "make_lm_action_features",
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
    "NextWordSamples",
    "build_word_forecaster",
    "clusterer_distance_matrix",
    "evaluate_word_forecast",
    "expected_next_word_sample_count",
    "fit_markov_prior_features",
    "make_markov_prior_feature_matrix",
    "make_next_word_samples",
    "NGramBackoffLanguageModel",
    "confidence_analysis",
    "error_analysis",
    "evaluate_language_model",
    "transition_entropy",
    "transition_quality_metrics",
    "word_distribution_metrics",
]
