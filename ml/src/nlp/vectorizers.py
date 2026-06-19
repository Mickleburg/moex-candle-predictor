"""Sentence vectorizers for candle-word sequences."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


@dataclass(frozen=True)
class VectorizerSpec:
    """Serializable vectorizer specification."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def label(self) -> str:
        if not self.params:
            return self.name
        suffix = ",".join(f"{key}={value}" for key, value in sorted(self.params.items()))
        return f"{self.name}({suffix})"


class SklearnTextVectorizer:
    """Count/TF-IDF wrapper with a common interface."""

    def __init__(self, kind: str, params: dict[str, Any]) -> None:
        token_pattern = params.pop("token_pattern", r"(?u)\b\w+\b")
        if kind == "count":
            self.model = CountVectorizer(token_pattern=token_pattern, **params)
        elif kind == "tfidf":
            self.model = TfidfVectorizer(token_pattern=token_pattern, **params)
        else:
            raise ValueError(f"Unsupported text vectorizer: {kind}")

    def fit_transform(self, sentences: list[str], token_lists: list[list[str]] | None = None):
        return self.model.fit_transform(sentences)

    def transform(self, sentences: list[str], token_lists: list[list[str]] | None = None):
        return self.model.transform(sentences)


class TfidfSvdVectorizer:
    """TF-IDF followed by latent semantic projection."""

    def __init__(self, params: dict[str, Any], random_state: int = 42) -> None:
        params = dict(params)
        self.n_components = int(params.pop("n_components", 32))
        self.random_state = random_state
        self.tfidf = TfidfVectorizer(
            token_pattern=params.pop("token_pattern", r"(?u)\b\w+\b"),
            **params,
        )
        self.svd: TruncatedSVD | None = None

    def fit_transform(self, sentences: list[str], token_lists: list[list[str]] | None = None):
        X = self.tfidf.fit_transform(sentences)
        n_components = min(self.n_components, max(1, min(X.shape) - 1))
        self.svd = TruncatedSVD(n_components=n_components, random_state=self.random_state)
        return self.svd.fit_transform(X)

    def transform(self, sentences: list[str], token_lists: list[list[str]] | None = None):
        if self.svd is None:
            raise ValueError("Vectorizer is not fitted")
        return self.svd.transform(self.tfidf.transform(sentences))


class CooccurrenceSvdVectorizer:
    """PPMI + SVD word embeddings summarized over a sentence window."""

    def __init__(self, params: dict[str, Any], random_state: int = 42) -> None:
        params = dict(params)
        self.embedding_dim = int(params.pop("embedding_dim", 16))
        self.context_window = int(params.pop("context_window", 2))
        self.min_count = int(params.pop("min_count", 1))
        pool = params.pop("pool", ("mean", "std", "last"))
        if isinstance(pool, str):
            pool = tuple(part for part in pool.replace("+", ",").split(",") if part)
        self.pool = tuple(pool)
        self.include_histogram = bool(params.pop("include_histogram", True))
        if params:
            raise ValueError(f"Unknown cooccurrence_svd params: {sorted(params)}")
        self.random_state = random_state
        self.vocab_: dict[str, int] = {}
        self.embeddings_: np.ndarray | None = None

    def fit_transform(self, sentences: list[str], token_lists: list[list[str]] | None = None) -> np.ndarray:
        token_lists = self._ensure_token_lists(sentences, token_lists)
        self._fit_embeddings(token_lists)
        return self.transform(sentences, token_lists)

    def transform(self, sentences: list[str], token_lists: list[list[str]] | None = None) -> np.ndarray:
        if self.embeddings_ is None:
            raise ValueError("Vectorizer is not fitted")
        token_lists = self._ensure_token_lists(sentences, token_lists)
        rows = [self._summarize(tokens) for tokens in token_lists]
        return np.vstack(rows) if rows else np.empty((0, self.output_dim_))

    @property
    def output_dim_(self) -> int:
        if self.embeddings_ is None:
            base_dim = self.embedding_dim
            vocab_dim = len(self.vocab_) if self.include_histogram else 0
            return base_dim * len(self.pool) + vocab_dim
        base_dim = self.embeddings_.shape[1] * len(self.pool)
        vocab_dim = len(self.vocab_) if self.include_histogram else 0
        return base_dim + vocab_dim

    def _ensure_token_lists(
        self,
        sentences: list[str],
        token_lists: list[list[str]] | None,
    ) -> list[list[str]]:
        if token_lists is not None:
            return token_lists
        return [sentence.split() for sentence in sentences]

    def _fit_embeddings(self, token_lists: list[list[str]]) -> None:
        counts = Counter(token for tokens in token_lists for token in tokens)
        vocab = sorted(token for token, count in counts.items() if count >= self.min_count)
        if not vocab:
            raise ValueError("No tokens survived min_count")
        self.vocab_ = {token: idx for idx, token in enumerate(vocab)}
        n_vocab = len(self.vocab_)

        cooc = np.zeros((n_vocab, n_vocab), dtype=float)
        for tokens in token_lists:
            ids = [self.vocab_[token] for token in tokens if token in self.vocab_]
            for center_pos, center_id in enumerate(ids):
                left = max(0, center_pos - self.context_window)
                right = min(len(ids), center_pos + self.context_window + 1)
                for context_pos in range(left, right):
                    if context_pos == center_pos:
                        continue
                    cooc[center_id, ids[context_pos]] += 1.0

        if n_vocab == 1 or cooc.sum() == 0:
            self.embeddings_ = np.zeros((n_vocab, 1), dtype=float)
            return

        total = cooc.sum()
        row_sum = cooc.sum(axis=1, keepdims=True)
        col_sum = cooc.sum(axis=0, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            pmi = np.log((cooc * total) / np.maximum(row_sum @ col_sum, 1e-12))
        ppmi = np.maximum(np.nan_to_num(pmi, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
        n_components = min(self.embedding_dim, max(1, min(ppmi.shape) - 1))
        svd = TruncatedSVD(n_components=n_components, random_state=self.random_state)
        self.embeddings_ = svd.fit_transform(ppmi)

    def _summarize(self, tokens: list[str]) -> np.ndarray:
        assert self.embeddings_ is not None
        ids = [self.vocab_[token] for token in tokens if token in self.vocab_]
        if ids:
            vectors = self.embeddings_[ids]
        else:
            vectors = np.zeros((1, self.embeddings_.shape[1]), dtype=float)

        parts: list[np.ndarray] = []
        for pool in self.pool:
            if pool == "mean":
                parts.append(vectors.mean(axis=0))
            elif pool == "sum":
                parts.append(vectors.sum(axis=0))
            elif pool == "std":
                parts.append(vectors.std(axis=0))
            elif pool == "max":
                parts.append(vectors.max(axis=0))
            elif pool == "last":
                parts.append(vectors[-1])
            else:
                raise ValueError(f"Unknown pooling mode: {pool}")

        if self.include_histogram:
            hist = np.zeros(len(self.vocab_), dtype=float)
            for idx in ids:
                hist[idx] += 1.0
            if hist.sum() > 0:
                hist /= hist.sum()
            parts.append(hist)

        return np.concatenate(parts)


def build_vectorizer(spec: VectorizerSpec, random_state: int = 42):
    """Instantiate a vectorizer from its spec."""

    name = spec.name.lower()
    params = dict(spec.params)
    if name in {"count", "tfidf"}:
        return SklearnTextVectorizer(name, params)
    if name == "tfidf_svd":
        return TfidfSvdVectorizer(params, random_state=random_state)
    if name == "cooccurrence_svd":
        return CooccurrenceSvdVectorizer(params, random_state=random_state)
    raise ValueError(f"Unknown vectorizer: {spec.name}")


def is_sparse_matrix(X: Any) -> bool:
    return sparse.issparse(X)
