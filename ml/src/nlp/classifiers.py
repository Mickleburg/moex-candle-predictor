"""Classifier factory for candle-language experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from scipy import sparse
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC, SVC


@dataclass(frozen=True)
class ClassifierSpec:
    """Serializable classifier specification."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def label(self) -> str:
        if not self.params:
            return self.name
        suffix = ",".join(f"{key}={value}" for key, value in sorted(self.params.items()))
        return f"{self.name}({suffix})"


def build_classifier(spec: ClassifierSpec, random_state: int = 42):
    """Instantiate a classifier from its spec."""

    name = spec.name.lower()
    params = dict(spec.params)

    if name == "logreg":
        return LogisticRegression(
            max_iter=int(params.pop("max_iter", 1000)),
            class_weight=params.pop("class_weight", "balanced"),
            random_state=random_state,
            **params,
        )
    if name == "ridge":
        return RidgeClassifier(
            class_weight=params.pop("class_weight", "balanced"),
            random_state=random_state,
            **params,
        )
    if name == "linear_svc":
        return LinearSVC(
            C=float(params.pop("C", 1.0)),
            class_weight=params.pop("class_weight", "balanced"),
            max_iter=int(params.pop("max_iter", 5000)),
            random_state=random_state,
            **params,
        )
    if name == "svc_rbf":
        return SVC(
            kernel="rbf",
            C=float(params.pop("C", 1.0)),
            gamma=params.pop("gamma", "scale"),
            class_weight=params.pop("class_weight", "balanced"),
            random_state=random_state,
            **params,
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=int(params.pop("n_estimators", 300)),
            max_depth=params.pop("max_depth", None),
            min_samples_leaf=int(params.pop("min_samples_leaf", 2)),
            class_weight=params.pop("class_weight", "balanced_subsample"),
            n_jobs=int(params.pop("n_jobs", -1)),
            random_state=random_state,
            **params,
        )
    if name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=int(params.pop("n_estimators", 300)),
            max_depth=params.pop("max_depth", None),
            min_samples_leaf=int(params.pop("min_samples_leaf", 2)),
            class_weight=params.pop("class_weight", "balanced"),
            n_jobs=int(params.pop("n_jobs", -1)),
            random_state=random_state,
            **params,
        )
    if name == "hist_gb":
        return HistGradientBoostingClassifier(
            max_iter=int(params.pop("max_iter", 200)),
            learning_rate=float(params.pop("learning_rate", 0.05)),
            max_leaf_nodes=int(params.pop("max_leaf_nodes", 31)),
            random_state=random_state,
            **params,
        )
    if name == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            objective=params.pop("objective", "multiclass"),
            num_class=int(params.pop("num_class", 3)),
            n_estimators=int(params.pop("n_estimators", 300)),
            learning_rate=float(params.pop("learning_rate", 0.03)),
            num_leaves=int(params.pop("num_leaves", 31)),
            class_weight=params.pop("class_weight", "balanced"),
            verbosity=int(params.pop("verbosity", -1)),
            random_state=random_state,
            n_jobs=int(params.pop("n_jobs", -1)),
            **params,
        )
    if name == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=params.pop("hidden_layer_sizes", (64, 32)),
            alpha=float(params.pop("alpha", 0.0005)),
            max_iter=int(params.pop("max_iter", 250)),
            early_stopping=bool(params.pop("early_stopping", True)),
            random_state=random_state,
            **params,
        )

    raise ValueError(f"Unknown classifier: {spec.name}")


def classifier_requires_dense(spec: ClassifierSpec) -> bool:
    """Return whether a classifier should receive a dense matrix."""

    return spec.name.lower() in {
        "random_forest",
        "extra_trees",
        "hist_gb",
        "svc_rbf",
        "mlp",
    }


def maybe_dense(X):
    """Convert sparse matrices to dense arrays when needed."""

    if sparse.issparse(X):
        return X.toarray()
    return X
