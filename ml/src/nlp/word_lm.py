"""N-gram/backoff language models for candle-word sequences."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


@dataclass(frozen=True)
class BeamCandidate:
    """Decoded sequence candidate with log probability."""

    sequence: tuple[int, ...]
    log_probability: float


class NGramBackoffLanguageModel:
    """Additive-smoothed n-gram LM with shorter-context backoff."""

    def __init__(self, order: int = 3, alpha: float = 0.1) -> None:
        if order < 1:
            raise ValueError("order must be >= 1")
        if alpha <= 0:
            raise ValueError("alpha must be > 0")
        self.order = int(order)
        self.alpha = float(alpha)
        self.n_words_: int | None = None
        self.counts_: list[dict[tuple[int, ...], np.ndarray]] = []

    def fit(
        self,
        words: Sequence[int],
        *,
        train_start: int = 0,
        train_end: int | None = None,
        n_words: int | None = None,
    ) -> "NGramBackoffLanguageModel":
        """Fit n-gram counts on one train range only."""

        word_array = np.asarray(words, dtype=int)
        train_end = len(word_array) if train_end is None else int(train_end)
        if train_start < 0 or train_end > len(word_array) or train_start >= train_end:
            raise ValueError("Invalid train range")
        if n_words is None:
            n_words = int(word_array[train_start:train_end].max()) + 1
        if n_words < 1:
            raise ValueError("n_words must be >= 1")
        self.n_words_ = int(n_words)

        counts: list[defaultdict[tuple[int, ...], np.ndarray]] = [
            defaultdict(lambda: np.zeros(self.n_words_, dtype=float)) for _ in range(self.order + 1)
        ]
        for pos in range(train_start, train_end):
            target = int(word_array[pos])
            if target < 0 or target >= self.n_words_:
                continue
            max_history = min(self.order, pos - train_start)
            for history_len in range(max_history + 1):
                history = tuple(int(item) for item in word_array[pos - history_len : pos]) if history_len else ()
                counts[history_len][history][target] += 1.0

        self.counts_ = [dict(item) for item in counts]
        return self

    def next_proba(self, context: Sequence[int]) -> np.ndarray:
        """Return P(next_word | context) with backoff and smoothing."""

        if self.n_words_ is None or not self.counts_:
            raise ValueError("Language model is not fitted")
        context_tuple = tuple(int(item) for item in context)
        for history_len in range(min(self.order, len(context_tuple)), -1, -1):
            history = context_tuple[-history_len:] if history_len else ()
            counts = self.counts_[history_len].get(history)
            if counts is not None:
                smoothed = counts + self.alpha
                return smoothed / smoothed.sum()
        return np.ones(self.n_words_, dtype=float) / self.n_words_

    def greedy_decode(self, context: Sequence[int], forecast_horizon: int) -> tuple[int, ...]:
        """Decode the most likely continuation greedily."""

        generated: list[int] = []
        running_context = list(int(item) for item in context)
        for _ in range(forecast_horizon):
            proba = self.next_proba(running_context)
            token = int(np.argmax(proba))
            generated.append(token)
            running_context.append(token)
        return tuple(generated)

    def beam_search(
        self,
        context: Sequence[int],
        forecast_horizon: int,
        *,
        beam_width: int = 5,
    ) -> list[BeamCandidate]:
        """Decode high-probability continuations with beam search."""

        if forecast_horizon < 1:
            raise ValueError("forecast_horizon must be >= 1")
        if beam_width < 1:
            raise ValueError("beam_width must be >= 1")

        beams = [(tuple(), 0.0)]
        base_context = [int(item) for item in context]
        for _ in range(forecast_horizon):
            expanded: list[tuple[tuple[int, ...], float]] = []
            for prefix, logp in beams:
                proba = self.next_proba([*base_context, *prefix])
                top = np.argsort(proba)[-beam_width:][::-1]
                for token in top:
                    token_int = int(token)
                    expanded.append(((*prefix, token_int), logp + float(np.log(max(proba[token_int], 1e-300)))))
            expanded.sort(key=lambda item: item[1], reverse=True)
            beams = expanded[:beam_width]
        return [BeamCandidate(sequence=seq, log_probability=logp) for seq, logp in beams]

    def predict(self, X_contexts: np.ndarray, forecast_horizon: int) -> np.ndarray:
        """Greedy free-running continuation for many contexts."""

        rows = [self.greedy_decode(row, forecast_horizon) for row in np.asarray(X_contexts, dtype=int)]
        return np.asarray(rows, dtype=int)

    def teacher_forced_probabilities(self, X_contexts: np.ndarray, Y_future_words: np.ndarray) -> list[np.ndarray]:
        """Return next-token probabilities conditioned on true previous targets."""

        X_contexts = np.asarray(X_contexts, dtype=int)
        Y_future_words = np.asarray(Y_future_words, dtype=int)
        horizon = Y_future_words.shape[1]
        probabilities = [np.zeros((len(X_contexts), int(self.n_words_)), dtype=float) for _ in range(horizon)]
        for row_idx, (context, future) in enumerate(zip(X_contexts, Y_future_words)):
            running_context = [int(item) for item in context]
            for step in range(horizon):
                probabilities[step][row_idx] = self.next_proba(running_context)
                running_context.append(int(future[step]))
        return probabilities

    def sequence_log_probability(self, context: Sequence[int], future: Sequence[int]) -> float:
        """Return log P(future sequence | context) with teacher forcing."""

        logp = 0.0
        running_context = [int(item) for item in context]
        for token in future:
            token_int = int(token)
            proba = self.next_proba(running_context)
            logp += float(np.log(max(proba[token_int], 1e-300)))
            running_context.append(token_int)
        return logp


def evaluate_language_model(
    model: NGramBackoffLanguageModel,
    X_contexts: np.ndarray,
    Y_future_words: np.ndarray,
    *,
    distance_matrix: np.ndarray | None = None,
    beam_width: int = 5,
) -> dict[str, Any]:
    """Evaluate token probabilities and free-running sequence continuation."""

    X_contexts = np.asarray(X_contexts, dtype=int)
    Y_future_words = np.asarray(Y_future_words, dtype=int)
    if Y_future_words.ndim != 2:
        raise ValueError("Y_future_words must be a 2D array")
    forecast_horizon = Y_future_words.shape[1]
    n_words = int(model.n_words_ or 0)
    if n_words < 1:
        raise ValueError("Model vocabulary is empty")

    greedy_pred = model.predict(X_contexts, forecast_horizon)
    teacher_probs = model.teacher_forced_probabilities(X_contexts, Y_future_words)
    sequence_logps = np.asarray(
        [model.sequence_log_probability(context, future) for context, future in zip(X_contexts, Y_future_words)],
        dtype=float,
    )
    sequence_nll = -sequence_logps
    token_nll = sequence_nll / max(1, forecast_horizon)
    mean_token_nll = float(token_nll.mean()) if len(token_nll) else 0.0

    per_horizon: list[dict[str, Any]] = []
    for step in range(forecast_horizon):
        true_step = Y_future_words[:, step]
        proba = teacher_probs[step]
        pred_step = np.argmax(proba, axis=1)
        item: dict[str, Any] = {
            "horizon": int(step + 1),
            "accuracy_at_1": float(accuracy_score(true_step, pred_step)),
            "top3_accuracy": _top_k_accuracy(true_step, proba, min(3, n_words)),
            "top5_accuracy": _top_k_accuracy(true_step, proba, min(5, n_words)),
            "macro_f1": float(f1_score(true_step, pred_step, labels=list(range(n_words)), average="macro", zero_division=0)),
            "mean_reciprocal_rank": _mean_reciprocal_rank(true_step, proba),
            "mean_token_nll": float(-np.log(np.maximum(proba[np.arange(len(true_step)), true_step], 1e-300)).mean()),
        }
        if distance_matrix is not None and distance_matrix.size:
            distances = distance_matrix[pred_step, true_step]
            nonzero = distance_matrix[distance_matrix > 0]
            tau = float(np.median(nonzero)) if len(nonzero) else 1.0
            item["mean_centroid_distance"] = float(distances.mean())
            item["mean_soft_similarity"] = float(np.exp(-distances / max(tau, 1e-12)).mean())
        per_horizon.append(item)

    result: dict[str, Any] = {
        "per_horizon": per_horizon,
        "mean_accuracy_at_1": float(np.mean([item["accuracy_at_1"] for item in per_horizon])) if per_horizon else 0.0,
        "mean_top3_accuracy": float(np.mean([item["top3_accuracy"] for item in per_horizon])) if per_horizon else 0.0,
        "mean_top5_accuracy": float(np.mean([item["top5_accuracy"] for item in per_horizon])) if per_horizon else 0.0,
        "mean_mrr": float(np.mean([item["mean_reciprocal_rank"] for item in per_horizon])) if per_horizon else 0.0,
        "sequence_exact_match": float(np.all(greedy_pred == Y_future_words, axis=1).mean()) if len(Y_future_words) else 0.0,
        "sequence_nll": float(sequence_nll.mean()) if len(sequence_nll) else 0.0,
        "mean_token_nll": mean_token_nll,
        "perplexity": float(np.exp(min(mean_token_nll, 700.0))),
        "beam_width": int(beam_width),
        "beam_contains_true_sequence": _beam_contains_rate(model, X_contexts, Y_future_words, beam_width=beam_width),
    }
    if distance_matrix is not None and distance_matrix.size:
        trajectory_distances = distance_matrix[greedy_pred, Y_future_words]
        result["average_centroid_trajectory_distance"] = float(trajectory_distances.mean()) if trajectory_distances.size else 0.0
        result["mean_soft_similarity"] = float(np.mean([item["mean_soft_similarity"] for item in per_horizon]))
        result["mean_centroid_distance"] = float(np.mean([item["mean_centroid_distance"] for item in per_horizon]))
    return result


def word_distribution_metrics(words: Sequence[int], n_words: int) -> dict[str, float]:
    """Summarize train-only vocabulary distribution quality."""

    word_array = np.asarray(words, dtype=int)
    counts = np.bincount(word_array[(word_array >= 0) & (word_array < n_words)], minlength=n_words).astype(float)
    total = counts.sum()
    if total <= 0:
        return {"entropy": 0.0, "normalized_entropy": 0.0, "dominant_share": 0.0, "nonempty_words": 0.0}
    proba = counts / total
    positive = proba[proba > 0]
    entropy = float(-np.sum(positive * np.log(positive)))
    normalized = entropy / np.log(n_words) if n_words > 1 else 0.0
    return {
        "entropy": entropy,
        "normalized_entropy": float(normalized),
        "dominant_share": float(proba.max()),
        "nonempty_words": float(np.count_nonzero(counts)),
    }


def transition_entropy(words: Sequence[int], n_words: int) -> float:
    """Return weighted entropy of one-step train transitions."""

    word_array = np.asarray(words, dtype=int)
    table = np.zeros((n_words, n_words), dtype=float)
    for current, nxt in zip(word_array[:-1], word_array[1:]):
        if 0 <= int(current) < n_words and 0 <= int(nxt) < n_words:
            table[int(current), int(nxt)] += 1.0
    row_sums = table.sum(axis=1)
    total = row_sums.sum()
    if total <= 0:
        return 0.0
    entropies = []
    weights = []
    for row, row_sum in zip(table, row_sums):
        if row_sum <= 0:
            continue
        proba = row / row_sum
        positive = proba[proba > 0]
        entropies.append(float(-np.sum(positive * np.log(positive))))
        weights.append(float(row_sum / total))
    return float(np.dot(entropies, weights)) if entropies else 0.0


def _top_k_accuracy(y_true: np.ndarray, proba: np.ndarray, k: int) -> float:
    top = np.argsort(proba, axis=1)[:, -k:]
    return float(np.mean([int(target) in row for target, row in zip(y_true, top)]))


def _mean_reciprocal_rank(y_true: np.ndarray, proba: np.ndarray) -> float:
    order = np.argsort(proba, axis=1)[:, ::-1]
    ranks = []
    for target, row in zip(y_true, order):
        positions = np.where(row == int(target))[0]
        ranks.append(1.0 / float(positions[0] + 1) if len(positions) else 0.0)
    return float(np.mean(ranks)) if ranks else 0.0


def _beam_contains_rate(
    model: NGramBackoffLanguageModel,
    X_contexts: np.ndarray,
    Y_future_words: np.ndarray,
    *,
    beam_width: int,
) -> float:
    hits = []
    for context, future in zip(X_contexts, Y_future_words):
        target = tuple(int(item) for item in future)
        beam = model.beam_search(context, len(target), beam_width=beam_width)
        hits.append(any(candidate.sequence == target for candidate in beam))
    return float(np.mean(hits)) if hits else 0.0
