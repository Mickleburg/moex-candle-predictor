"""Leakage-safe LM-derived features for downstream action classifiers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .word_lm import NGramBackoffLanguageModel


@dataclass(frozen=True)
class LMActionFeatures:
    """Feature matrix and column names for action samples."""

    X: np.ndarray
    names: list[str]


def make_lm_action_features(
    *,
    word_ids: Sequence[int],
    target_indices: Sequence[int],
    context_size: int,
    model: NGramBackoffLanguageModel,
    distance_matrix: np.ndarray | None = None,
    include_probabilities: bool = False,
    include_topn: int = 3,
    beam_horizon: int = 3,
    beam_width: int = 3,
) -> LMActionFeatures:
    """Build next-word LM features using only words known at each target index.

    The function does not accept future target words by design. For a sample
    ending at ``target_idx=t``, the LM context is
    ``words[t-context_size+1]..words[t]``.
    """

    if context_size < 1:
        raise ValueError("context_size must be >= 1")
    if model.n_words_ is None:
        raise ValueError("Language model is not fitted")

    word_array = np.asarray(word_ids, dtype=int)
    target_indices = np.asarray(target_indices, dtype=int)
    n_words = int(model.n_words_)
    scalar_rows: list[list[float]] = []
    proba_rows: list[np.ndarray] = []

    for target_idx in target_indices:
        start = int(target_idx) - context_size + 1
        if start < 0:
            raise ValueError("LM context would start before the word sequence")
        context = word_array[start : int(target_idx) + 1]
        if len(context) != context_size:
            raise ValueError("LM context length mismatch")
        if np.any(context < 0):
            raise ValueError("LM context contains unassigned word IDs")

        proba = model.next_proba(context)
        if not np.all(np.isfinite(proba)) or not np.isclose(proba.sum(), 1.0):
            raise ValueError("Invalid LM probability distribution")
        current_word = int(context[-1])
        order = np.argsort(proba)[::-1]
        top = order[: max(2, include_topn)]
        top1 = int(top[0])
        top2 = int(top[1]) if len(top) > 1 else top1
        top3 = order[: min(3, n_words)]
        top3_mass = float(proba[top3].sum())
        entropy = float(-np.sum(proba[proba > 0] * np.log(proba[proba > 0])))
        expected_distance = 0.0
        if distance_matrix is not None and distance_matrix.size and 0 <= current_word < distance_matrix.shape[0]:
            expected_distance = float(np.dot(proba, distance_matrix[current_word, :n_words]))

        beam = model.beam_search(context, max(1, beam_horizon), beam_width=beam_width)
        beam_best = beam[0].log_probability if beam else 0.0
        beam_second = beam[1].log_probability if len(beam) > 1 else beam_best
        mean_step_entropy = _rollout_mean_entropy(model, context, max(1, beam_horizon))

        row = [
            float(proba[top1]),
            float(proba[top2]),
            top3_mass,
            entropy,
            float(proba[top1] - proba[top2]),
            float(top1 / max(1, n_words - 1)),
            expected_distance,
            float(proba[current_word]) if 0 <= current_word < n_words else 0.0,
            float(beam_best),
            float(beam_second),
            float(beam_best - beam_second),
            mean_step_entropy,
        ]
        for rank in range(include_topn):
            word = int(order[rank]) if rank < len(order) else 0
            row.extend([float(word / max(1, n_words - 1)), float(proba[word])])
        scalar_rows.append(row)
        if include_probabilities:
            proba_rows.append(proba.astype(float))

    names = [
        "lm_top1_prob",
        "lm_top2_prob",
        "lm_top3_mass",
        "lm_entropy",
        "lm_margin_top1_top2",
        "lm_predicted_next_word_id_norm",
        "lm_expected_centroid_distance_from_current",
        "lm_self_transition_probability",
        "lm_beam_best_logprob",
        "lm_beam_second_logprob",
        "lm_beam_margin",
        "lm_mean_step_entropy",
    ]
    for rank in range(include_topn):
        names.extend([f"lm_top{rank + 1}_word_id_norm", f"lm_top{rank + 1}_prob"])

    X = np.asarray(scalar_rows, dtype=float)
    if include_probabilities:
        X = np.hstack([X, np.vstack(proba_rows)])
        names.extend([f"lm_word_proba_{idx}" for idx in range(n_words)])
    if not np.all(np.isfinite(X)):
        raise ValueError("LM action features contain non-finite values")
    return LMActionFeatures(X=X, names=names)


def _rollout_mean_entropy(model: NGramBackoffLanguageModel, context: np.ndarray, horizon: int) -> float:
    entropies = []
    running = [int(item) for item in context]
    for _ in range(horizon):
        proba = model.next_proba(running)
        positive = proba[proba > 0]
        entropies.append(float(-np.sum(positive * np.log(positive))))
        running.append(int(np.argmax(proba)))
    return float(np.mean(entropies)) if entropies else 0.0
