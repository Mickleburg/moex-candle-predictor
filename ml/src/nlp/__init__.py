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
from .action_features import LMActionFeatures, make_continuous_past_features, make_lm_action_features, standardize_by_train
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
from .targets import (
    ActionTargetResult,
    ActionTargetSpec,
    make_research_action_targets,
    past_return_volatility,
    target_analysis,
    triple_barrier_details,
)

__all__ = [
    "ACTION_LABELS",
    "LMActionFeatures",
    "SentenceSamples",
    "candle_shape_matrix",
    "label_distribution",
    "make_action_labels",
    "make_continuous_past_features",
    "make_lm_action_features",
    "make_sentence_samples",
    "standardize_by_train",
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
    "ActionTargetResult",
    "ActionTargetSpec",
    "make_research_action_targets",
    "past_return_volatility",
    "target_analysis",
    "triple_barrier_details",
]
