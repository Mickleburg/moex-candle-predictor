"""Next candle-word forecasting utilities."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


@dataclass(frozen=True)
class NextWordSamples:
    """Context word windows and future word targets."""

    X_contexts: np.ndarray
    Y_future_words: np.ndarray
    sample_indices: np.ndarray

    @property
    def size(self) -> int:
        return int(len(self.sample_indices))


def make_next_word_samples(
    words: Sequence[int],
    split_start: int,
    split_end: int,
    context_size: int,
    forecast_horizon: int,
) -> NextWordSamples:
    """Build strict within-split next-word samples.

    Input is ``words[t-context_size+1]..words[t]`` and target is
    ``words[t+1]..words[t+forecast_horizon]``. No input contains a future word,
    and each target sequence stays inside the split.
    """

    if context_size < 1:
        raise ValueError("context_size must be >= 1")
    if forecast_horizon < 1:
        raise ValueError("forecast_horizon must be >= 1")
    if split_start < 0 or split_end > len(words) or split_start >= split_end:
        raise ValueError("Invalid split range")

    word_array = np.asarray(words, dtype=int)
    contexts: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    indices: list[int] = []

    first_t = split_start + context_size - 1
    last_t_exclusive = split_end - forecast_horizon
    for t_idx in range(first_t, last_t_exclusive):
        context = word_array[t_idx - context_size + 1 : t_idx + 1]
        future = word_array[t_idx + 1 : t_idx + forecast_horizon + 1]
        if len(context) != context_size or len(future) != forecast_horizon:
            continue
        contexts.append(context)
        targets.append(future)
        indices.append(t_idx)

    return NextWordSamples(
        X_contexts=np.vstack(contexts) if contexts else np.empty((0, context_size), dtype=int),
        Y_future_words=np.vstack(targets) if targets else np.empty((0, forecast_horizon), dtype=int),
        sample_indices=np.asarray(indices, dtype=int),
    )


def expected_next_word_sample_count(split_len: int, context_size: int, forecast_horizon: int) -> int:
    return max(0, int(split_len) - int(context_size) - int(forecast_horizon) + 1)


class PersistenceWordForecaster:
    """Predict every future word as the latest observed word."""

    label = "persistence"

    def fit(self, X: np.ndarray, Y: np.ndarray, n_words: int) -> "PersistenceWordForecaster":
        self.forecast_horizon_ = Y.shape[1]
        self.n_words_ = n_words
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.repeat(X[:, [-1]], self.forecast_horizon_, axis=1)


class UnigramWordForecaster:
    """Predict the most frequent train future word."""

    label = "unigram"

    def fit(self, X: np.ndarray, Y: np.ndarray, n_words: int) -> "UnigramWordForecaster":
        self.forecast_horizon_ = Y.shape[1]
        self.n_words_ = n_words
        counts = np.bincount(Y.ravel(), minlength=n_words)
        self.prediction_ = int(np.argmax(counts))
        self.proba_ = counts / counts.sum() if counts.sum() else np.ones(n_words) / n_words
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full((len(X), self.forecast_horizon_), self.prediction_, dtype=int)

    def predict_proba(self, X: np.ndarray) -> list[np.ndarray]:
        return [np.tile(self.proba_, (len(X), 1)) for _ in range(self.forecast_horizon_)]


class MarkovWordForecaster:
    """Direct Markov baseline from the last context word to each future step."""

    def __init__(self, order: int = 1) -> None:
        if order != 1:
            raise ValueError("Only Markov order=1 is implemented for next-word forecasting")
        self.order = order
        self.label = f"markov{order}"

    def fit(self, X: np.ndarray, Y: np.ndarray, n_words: int) -> "MarkovWordForecaster":
        self.forecast_horizon_ = Y.shape[1]
        self.n_words_ = n_words
        self.default_ = np.bincount(Y.ravel(), minlength=n_words).astype(float)
        self.default_ = self.default_ / self.default_.sum() if self.default_.sum() else np.ones(n_words) / n_words
        self.tables_: list[np.ndarray] = []
        last_words = X[:, -1]
        for step in range(self.forecast_horizon_):
            table = np.zeros((n_words, n_words), dtype=float)
            for last_word, target in zip(last_words, Y[:, step]):
                table[int(last_word), int(target)] += 1.0
            row_sums = table.sum(axis=1, keepdims=True)
            table = np.divide(table, row_sums, out=np.zeros_like(table), where=row_sums > 0)
            self.tables_.append(table)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        last_words = X[:, -1].astype(int)
        preds = np.zeros((len(X), self.forecast_horizon_), dtype=int)
        for step, table in enumerate(self.tables_):
            rows = table[last_words]
            empty = rows.sum(axis=1) == 0
            rows = rows.copy()
            rows[empty] = self.default_
            preds[:, step] = np.argmax(rows, axis=1)
        return preds

    def predict_proba(self, X: np.ndarray) -> list[np.ndarray]:
        last_words = X[:, -1].astype(int)
        probabilities = []
        for table in self.tables_:
            rows = table[last_words].copy()
            empty = rows.sum(axis=1) == 0
            rows[empty] = self.default_
            probabilities.append(rows)
        return probabilities


class TextLogisticWordForecaster:
    """Per-horizon logistic classifiers over count or TF-IDF context n-grams."""

    def __init__(self, kind: str = "tfidf", ngram_range: tuple[int, int] = (1, 3), max_features: int = 5000) -> None:
        self.kind = kind
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.label = f"{kind}_logreg"

    def fit(self, X: np.ndarray, Y: np.ndarray, n_words: int) -> "TextLogisticWordForecaster":
        self.forecast_horizon_ = Y.shape[1]
        self.n_words_ = n_words
        vectorizer_cls = TfidfVectorizer if self.kind == "tfidf" else CountVectorizer
        self.vectorizer_ = vectorizer_cls(
            token_pattern=r"(?u)\b\w+\b",
            ngram_range=self.ngram_range,
            max_features=self.max_features,
        )
        X_text = contexts_to_sentences(X)
        X_vec = self.vectorizer_.fit_transform(X_text)
        self.models_: list[LogisticRegression] = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for step in range(self.forecast_horizon_):
                model = LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                )
                model.fit(X_vec, Y[:, step])
                self.models_.append(model)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_vec = self.vectorizer_.transform(contexts_to_sentences(X))
        return np.column_stack([model.predict(X_vec) for model in self.models_]).astype(int)

    def predict_proba(self, X: np.ndarray) -> list[np.ndarray]:
        X_vec = self.vectorizer_.transform(contexts_to_sentences(X))
        probabilities = []
        for model in self.models_:
            proba = model.predict_proba(X_vec)
            full = np.zeros((len(X), self.n_words_), dtype=float)
            for src_idx, cls in enumerate(model.classes_):
                full[:, int(cls)] = proba[:, src_idx]
            probabilities.append(full)
        return probabilities


def contexts_to_sentences(X: np.ndarray) -> list[str]:
    return [" ".join(f"w{int(word):03d}" for word in row) for row in X]


def build_word_forecaster(name: str):
    if name == "persistence":
        return PersistenceWordForecaster()
    if name == "unigram":
        return UnigramWordForecaster()
    if name == "markov1":
        return MarkovWordForecaster(order=1)
    if name == "tfidf_logreg":
        return TextLogisticWordForecaster(kind="tfidf")
    if name == "count_logreg":
        return TextLogisticWordForecaster(kind="count")
    raise ValueError(f"Unknown word forecaster: {name}")


def clusterer_distance_matrix(clusterer: Any) -> np.ndarray:
    """Return train-fitted centroid distances between word IDs."""

    n_words = int(clusterer.n_words_)
    distances = np.zeros((n_words, n_words), dtype=float)
    centroids = getattr(clusterer, "centroids_", None)
    centroid_words = getattr(clusterer, "centroid_words_", None)
    if centroids is None or centroid_words is None or len(centroid_words) == 0:
        return distances

    known_distances = ((centroids[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2) ** 0.5
    max_distance = float(known_distances.max()) if known_distances.size else 0.0
    fill = max_distance if max_distance > 0 else 1.0
    distances.fill(fill)
    for i, word_i in enumerate(centroid_words):
        for j, word_j in enumerate(centroid_words):
            distances[int(word_i), int(word_j)] = float(known_distances[i, j])
    np.fill_diagonal(distances, 0.0)
    return distances


def evaluate_word_forecast(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    n_words: int,
    distance_matrix: np.ndarray | None = None,
    probabilities: list[np.ndarray] | None = None,
    nearest_n: int = 3,
) -> dict[str, Any]:
    """Compute exact and similarity-aware next-word metrics."""

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}")
    forecast_horizon = y_true.shape[1] if y_true.ndim == 2 else 0
    per_horizon: list[dict[str, Any]] = []
    for step in range(forecast_horizon):
        true_step = y_true[:, step]
        pred_step = y_pred[:, step]
        item: dict[str, Any] = {
            "horizon": int(step + 1),
            "accuracy": float(accuracy_score(true_step, pred_step)),
            "macro_f1": float(
                f1_score(true_step, pred_step, labels=list(range(n_words)), average="macro", zero_division=0)
            ),
        }
        if probabilities is not None:
            item["top3_accuracy"] = _top_k_accuracy(true_step, probabilities[step], k=min(3, n_words))
        if distance_matrix is not None and distance_matrix.size:
            distances = distance_matrix[pred_step, true_step]
            nonzero = distance_matrix[distance_matrix > 0]
            tau = float(np.median(nonzero)) if len(nonzero) else 1.0
            similarity = np.exp(-distances / max(tau, 1e-12))
            item["mean_centroid_distance"] = float(distances.mean())
            item["mean_soft_similarity"] = float(similarity.mean())
            item["soft_similarity_tau"] = tau
            item[f"within_nearest_{nearest_n}_accuracy"] = _within_nearest_n_accuracy(
                true_step,
                pred_step,
                distance_matrix,
                nearest_n,
            )
        per_horizon.append(item)

    result = {
        "sequence_exact_match": float(np.all(y_true == y_pred, axis=1).mean()) if len(y_true) else 0.0,
        "per_horizon": per_horizon,
    }
    if per_horizon:
        result["mean_accuracy"] = float(np.mean([item["accuracy"] for item in per_horizon]))
        result["mean_macro_f1"] = float(np.mean([item["macro_f1"] for item in per_horizon]))
        if "mean_soft_similarity" in per_horizon[0]:
            result["mean_soft_similarity"] = float(np.mean([item["mean_soft_similarity"] for item in per_horizon]))
            result["mean_centroid_distance"] = float(np.mean([item["mean_centroid_distance"] for item in per_horizon]))
    return result


def _top_k_accuracy(y_true: np.ndarray, proba: np.ndarray, k: int) -> float:
    top = np.argsort(proba, axis=1)[:, -k:]
    return float(np.mean([int(target) in row for target, row in zip(y_true, top)]))


def _within_nearest_n_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    distance_matrix: np.ndarray,
    nearest_n: int,
) -> float:
    nearest = np.argsort(distance_matrix, axis=1)[:, : max(1, nearest_n)]
    return float(np.mean([int(pred) in nearest[int(true)] for true, pred in zip(y_true, y_pred)]))
