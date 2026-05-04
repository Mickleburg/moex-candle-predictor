"""Candle shape clustering for candle words."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.cluster import AgglomerativeClustering, DBSCAN, HDBSCAN, KMeans, MiniBatchKMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class ClusterSpec:
    """Serializable clustering method specification."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def label(self) -> str:
        if not self.params:
            return self.name
        suffix = ",".join(f"{key}={value}" for key, value in sorted(self.params.items()))
        return f"{self.name}({suffix})"


class CandleClusterer:
    """Fit candle-shape clusters and expose them as word IDs."""

    def __init__(
        self,
        spec: ClusterSpec,
        random_state: int = 42,
        scale: bool = True,
    ) -> None:
        self.spec = spec
        self.random_state = random_state
        self.scale = scale
        self.scaler_: StandardScaler | None = None
        self.model_: Any | None = None
        self.raw_to_word_: dict[int, int] = {}
        self.noise_word_: int | None = None
        self.centroids_: np.ndarray | None = None
        self.centroid_words_: np.ndarray | None = None
        self.train_labels_: np.ndarray | None = None
        self.quality_: dict[str, float] = {}

    def fit(self, X: np.ndarray) -> "CandleClusterer":
        """Fit clustering on train candle shapes only."""

        X_scaled = self._fit_scale(X)
        name = self.spec.name.lower()
        params = dict(self.spec.params)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if name == "kmeans":
                n_clusters = int(params.pop("n_clusters", params.pop("k", 20)))
                self.model_ = KMeans(
                    n_clusters=n_clusters,
                    random_state=self.random_state,
                    **params,
                )
                raw_labels = self.model_.fit_predict(X_scaled)
            elif name == "minibatch_kmeans":
                n_clusters = int(params.pop("n_clusters", params.pop("k", 20)))
                self.model_ = MiniBatchKMeans(
                    n_clusters=n_clusters,
                    random_state=self.random_state,
                    **params,
                )
                raw_labels = self.model_.fit_predict(X_scaled)
            elif name == "agglomerative":
                n_clusters = int(params.pop("n_clusters", params.pop("k", 20)))
                self.model_ = AgglomerativeClustering(n_clusters=n_clusters, **params)
                raw_labels = self.model_.fit_predict(X_scaled)
            elif name == "gmm":
                n_components = int(params.pop("n_components", params.pop("k", 20)))
                self.model_ = GaussianMixture(
                    n_components=n_components,
                    random_state=self.random_state,
                    **params,
                )
                raw_labels = self.model_.fit_predict(X_scaled)
            elif name == "dbscan":
                self.model_ = DBSCAN(**params)
                raw_labels = self.model_.fit_predict(X_scaled)
            elif name == "hdbscan":
                self.model_ = HDBSCAN(**params)
                raw_labels = self.model_.fit_predict(X_scaled)
            else:
                raise ValueError(f"Unknown clusterer: {self.spec.name}")

        self.train_labels_ = self._normalize_fit_labels(raw_labels)
        self._fit_centroids(X_scaled, self.train_labels_)
        self.quality_ = self._cluster_quality(X_scaled, self.train_labels_)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict word IDs for new candle shapes."""

        if self.model_ is None:
            raise ValueError("Clusterer is not fitted")
        X_scaled = self._transform_scale(X)
        name = self.spec.name.lower()

        if name in {"kmeans", "minibatch_kmeans", "gmm"} and hasattr(self.model_, "predict"):
            raw_labels = self.model_.predict(X_scaled)
            return self._normalize_predict_labels(raw_labels, X_scaled)

        return self._nearest_centroid_words(X_scaled)

    def labels_to_words(self, labels: np.ndarray) -> list[str]:
        return [f"w{int(label):03d}" for label in labels]

    @property
    def n_words_(self) -> int:
        if self.train_labels_ is None or len(self.train_labels_) == 0:
            return 0
        return int(np.max(self.train_labels_) + 1)

    def _fit_scale(self, X: np.ndarray) -> np.ndarray:
        if not self.scale:
            return np.asarray(X, dtype=float)
        self.scaler_ = StandardScaler()
        return self.scaler_.fit_transform(np.asarray(X, dtype=float))

    def _transform_scale(self, X: np.ndarray) -> np.ndarray:
        if not self.scale:
            return np.asarray(X, dtype=float)
        if self.scaler_ is None:
            raise ValueError("Scaler is not fitted")
        return self.scaler_.transform(np.asarray(X, dtype=float))

    def _normalize_fit_labels(self, raw_labels: np.ndarray) -> np.ndarray:
        raw_labels = np.asarray(raw_labels, dtype=int)
        non_noise = sorted(int(label) for label in np.unique(raw_labels) if int(label) != -1)
        if not non_noise:
            raise ValueError(f"{self.spec.label} produced only noise labels")

        self.raw_to_word_ = {raw: idx for idx, raw in enumerate(non_noise)}
        normalized = np.empty_like(raw_labels, dtype=int)
        for idx, raw in enumerate(raw_labels):
            raw_int = int(raw)
            if raw_int == -1:
                if self.noise_word_ is None:
                    self.noise_word_ = len(self.raw_to_word_)
                normalized[idx] = self.noise_word_
            else:
                normalized[idx] = self.raw_to_word_[raw_int]
        return normalized

    def _normalize_predict_labels(self, raw_labels: np.ndarray, X_scaled: np.ndarray) -> np.ndarray:
        normalized = np.empty(len(raw_labels), dtype=int)
        unknown_positions: list[int] = []
        for idx, raw in enumerate(np.asarray(raw_labels, dtype=int)):
            if int(raw) in self.raw_to_word_:
                normalized[idx] = self.raw_to_word_[int(raw)]
            else:
                unknown_positions.append(idx)
        if unknown_positions:
            nearest = self._nearest_centroid_words(X_scaled[unknown_positions])
            normalized[np.asarray(unknown_positions, dtype=int)] = nearest
        return normalized

    def _fit_centroids(self, X_scaled: np.ndarray, labels: np.ndarray) -> None:
        words = sorted(int(label) for label in np.unique(labels) if int(label) != self.noise_word_)
        if not words:
            raise ValueError("No non-noise clusters available for centroid assignment")
        centroids = []
        for word in words:
            centroids.append(X_scaled[labels == word].mean(axis=0))
        self.centroids_ = np.vstack(centroids)
        self.centroid_words_ = np.asarray(words, dtype=int)

    def _nearest_centroid_words(self, X_scaled: np.ndarray) -> np.ndarray:
        if self.centroids_ is None or self.centroid_words_ is None:
            raise ValueError("Centroids are not fitted")
        distances = ((X_scaled[:, None, :] - self.centroids_[None, :, :]) ** 2).sum(axis=2)
        nearest = np.argmin(distances, axis=1)
        return self.centroid_words_[nearest]

    def _cluster_quality(self, X_scaled: np.ndarray, labels: np.ndarray) -> dict[str, float]:
        valid = np.ones(len(labels), dtype=bool)
        if self.noise_word_ is not None:
            valid = labels != self.noise_word_

        labels_valid = labels[valid]
        X_valid = X_scaled[valid]
        result = {
            "n_words": float(self.n_words_),
            "noise_ratio": float(1.0 - valid.mean()),
        }
        unique = np.unique(labels_valid)
        if len(unique) < 2 or len(X_valid) <= len(unique):
            result.update(
                {
                    "silhouette": float("nan"),
                    "davies_bouldin": float("nan"),
                    "calinski_harabasz": float("nan"),
                }
            )
            return result

        sample_size = min(5000, len(X_valid))
        if sample_size < len(X_valid):
            rng = np.random.default_rng(self.random_state)
            sample_idx = rng.choice(len(X_valid), size=sample_size, replace=False)
            X_score = X_valid[sample_idx]
            y_score = labels_valid[sample_idx]
        else:
            X_score = X_valid
            y_score = labels_valid

        result["silhouette"] = float(silhouette_score(X_score, y_score))
        result["davies_bouldin"] = float(davies_bouldin_score(X_score, y_score))
        result["calinski_harabasz"] = float(calinski_harabasz_score(X_score, y_score))
        return result
