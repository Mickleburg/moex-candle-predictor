"""
Candle embedding features following Poženel & Lavbič (2019), arXiv:1902.08684.

Pipeline:
    normalize_ohlc  →  KMeans vocabulary  →  word IDs
         ↓
    co-occurrence SVD embeddings  (replaces gensim Word2Vec — no C extension
    required, captures the same "similar patterns in similar contexts" objective)
         ↓
    context_features: mean of nm previous embeddings per timestep

gensim is unavailable under Python 3.14 (no pre-built wheel). SVD on the
co-occurrence matrix is mathematically equivalent to GloVe/LSA and captures
the same semantic structure without a neural training loop.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize


# ── OHLC normalisation ───────────────────────────────────────────────────────

def normalize_ohlc(df: pd.DataFrame) -> np.ndarray:
    """Paper Eq.(2): represent each candle as (1, H/O, L/O, C/O).

    The constant 1 has no clustering value, so we return the 3-dim vector
    (H/O, L/O, C/O) — numerically identical to the paper's clustering input.
    Rows where open == 0 are set to (1, 1, 1) (neutral candle).
    """
    O = df["open"].astype(float).values
    H = df["high"].astype(float).values
    L = df["low"].astype(float).values
    C = df["close"].astype(float).values

    safe_O = np.where(np.abs(O) < 1e-12, np.nan, O)
    ratio_H = H / safe_O
    ratio_L = L / safe_O
    ratio_C = C / safe_O

    matrix = np.column_stack([ratio_H, ratio_L, ratio_C])
    matrix = np.nan_to_num(matrix, nan=1.0, posinf=1.0, neginf=1.0)
    return matrix.astype(float)


# ── K-Means vocabulary ───────────────────────────────────────────────────────

def fit_candle_vocabulary(X_norm: np.ndarray, nw: int = 30, seed: int = 42) -> KMeans:
    """K-Means on normalised OHLC shapes → vocabulary of nw candle patterns."""
    km = KMeans(n_clusters=nw, random_state=seed, n_init=10)
    km.fit(X_norm)
    return km


def candles_to_words(X_norm: np.ndarray, km: KMeans) -> np.ndarray:
    """Assign each candle a word ID (integer cluster label)."""
    return km.predict(X_norm).astype(int)


# ── Co-occurrence SVD embeddings ─────────────────────────────────────────────

def build_cooccurrence_embeddings(
    words: np.ndarray,
    n_words: int,
    nv: int = 32,
    window: int = 10,
    seed: int = 42,
) -> np.ndarray:
    """Build dense word embeddings via co-occurrence matrix + TruncatedSVD.

    Mathematically equivalent to GloVe/LSA: candle patterns that appear in
    similar contexts (windows of preceding/following patterns) get similar
    embeddings. Replaces gensim Word2Vec (skip-gram) which cannot be built
    on Python 3.14 due to missing pre-built Cython wheels.

    Args:
        words:   integer array of word IDs, shape (N,)
        n_words: vocabulary size (max word ID + 1)
        nv:      embedding dimensionality
        window:  context window on each side
        seed:    random state for SVD

    Returns:
        embeddings: ndarray of shape (n_words, nv) — one vector per word
    """
    words = np.asarray(words, dtype=int)
    n = len(words)

    # Build co-occurrence matrix (symmetric, context window on both sides)
    row_list: list[int] = []
    col_list: list[int] = []
    data_list: list[float] = []

    for i in range(n):
        w = int(words[i])
        lo = max(0, i - window)
        hi = min(n, i + window + 1)
        for j in range(lo, hi):
            if j != i:
                dist = abs(i - j)
                # Harmonic weighting: closer neighbours count more
                weight = 1.0 / dist
                c = int(words[j])
                row_list.append(w)
                col_list.append(c)
                data_list.append(weight)

    C = csr_matrix(
        (data_list, (row_list, col_list)),
        shape=(n_words, n_words),
        dtype=float,
    )

    # Log-scale (PPMI-like) smoothing to reduce dominance of frequent co-occurrences
    C = C.log1p()

    n_components = min(nv, n_words - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    embeddings = svd.fit_transform(C)  # shape: (n_words, n_components)

    # L2-normalize each word vector for stable dot products
    embeddings = normalize(embeddings, norm="l2")

    # Pad to nv if needed (when n_words < nv + 1)
    if embeddings.shape[1] < nv:
        pad = np.zeros((n_words, nv - embeddings.shape[1]))
        embeddings = np.hstack([embeddings, pad])

    return embeddings.astype(float)


# ── Context feature matrix ────────────────────────────────────────────────────

def make_context_features(
    words: np.ndarray,
    embeddings: np.ndarray,
    nm: int = 15,
) -> tuple[np.ndarray, list[str]]:
    """Paper Eq.(17): context_t = mean of nm previous word embeddings.

    At timestep i, the context is embeddings[words[i-nm : i]] averaged.
    First nm rows are zero-padded (no history yet).

    Args:
        words:      integer array of word IDs, shape (N,)
        embeddings: matrix of shape (n_words, nv)
        nm:         context window size (number of past candles to average)

    Returns:
        (matrix, feature_names):
            matrix      shape (N, nv)  — one context vector per candle
            feature_names  list of 'w2v_{k}' strings
    """
    words = np.asarray(words, dtype=int)
    n = len(words)
    nv = embeddings.shape[1]
    matrix = np.zeros((n, nv), dtype=float)

    for i in range(nm, n):
        # Collect the nm candles immediately preceding position i
        window_words = words[i - nm: i]
        vecs = embeddings[window_words]   # shape: (nm, nv)
        matrix[i] = vecs.mean(axis=0)

    names = [f"w2v_{k}" for k in range(nv)]
    return matrix, names


# ── Combined feature builder ──────────────────────────────────────────────────

def make_w2v_context_features(
    df: pd.DataFrame,
    *,
    train_end: int,
    nw: int = 30,
    nv: int = 32,
    window: int = 10,
    nm: int = 15,
    seed: int = 42,
) -> tuple[np.ndarray, list[str], dict]:
    """End-to-end: raw candles → context embedding features.

    Fits vocabulary and embeddings on train portion only (no leakage).

    Args:
        df:        full DataFrame with open/high/low/close columns
        train_end: index up to which training data goes (exclusive)
        nw:        number of K-Means clusters (vocabulary size)
        nv:        SVD embedding dimensions
        window:    co-occurrence window
        nm:        context window (number of past words to average)
        seed:      random seed

    Returns:
        (matrix, feature_names, info_dict)
        matrix: shape (len(df), nv) — context vectors for all rows
    """
    X_norm = normalize_ohlc(df)

    # Fit vocabulary on train only
    X_train_norm = X_norm[:train_end]
    km = fit_candle_vocabulary(X_train_norm, nw=nw, seed=seed)

    # Assign words to all candles (using train-fitted vocabulary)
    words = candles_to_words(X_norm, km)

    # Build embeddings from train word sequence only
    train_words = words[:train_end]
    embeddings = build_cooccurrence_embeddings(
        train_words, n_words=nw, nv=nv, window=window, seed=seed
    )

    # Build context features for all rows (first nm rows will be zero)
    matrix, names = make_context_features(words, embeddings, nm=nm)

    info = {
        "nw": nw,
        "nv": nv,
        "window": window,
        "nm": nm,
        "vocab_size": nw,
        "embedding_shape": list(embeddings.shape),
    }
    return matrix, names, info
