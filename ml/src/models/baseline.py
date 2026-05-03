"""Baseline models for comparison."""

import numpy as np
from scipy.stats import mode
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..utils.io import load_joblib, save_joblib


class MajorityClassifier:
    """Majority class baseline.
    
    Always predicts the most frequent class from training data.
    """
    
    def __init__(self, n_classes: int | None = None):
        """Initialize majority classifier."""
        self.n_classes = n_classes
        self.most_frequent_class_: int = None
        self.class_probabilities_: np.ndarray = None
        self.is_fitted_ = False
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "MajorityClassifier":
        """Fit majority classifier.
        
        Args:
            X: Feature array (ignored).
            y: Target labels.
            
        Returns:
            Self.
        """
        # Find most frequent class (ignore -1 which is invalid token)
        valid_mask = y != -1
        valid_y = y[valid_mask]
        
        if len(valid_y) == 0:
            raise ValueError("No valid labels found (all are -1)")
        
        result = mode(valid_y, keepdims=True)
        self.most_frequent_class_ = result.mode[0]
        
        # Compute class probabilities
        unique_classes, counts = np.unique(valid_y, return_counts=True)
        n_classes = self.n_classes or (max(unique_classes) + 1)
        self.class_probabilities_ = np.zeros(n_classes)
        for cls, count in zip(unique_classes, counts):
            if 0 <= cls < n_classes:
                self.class_probabilities_[cls] = count / len(valid_y)
        
        self.is_fitted_ = True
        print(f"Majority class: {self.most_frequent_class_}")
        print(f"Class distribution: {self.class_probabilities_}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict most frequent class.
        
        Args:
            X: Feature array (ignored).
            
        Returns:
            Array of predicted class labels.
        """
        if not self.is_fitted_:
            raise ValueError("Model not fitted. Call fit() first.")
        
        n_samples = X.shape[0]
        return np.full(n_samples, self.most_frequent_class_)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.
        
        Args:
            X: Feature array (ignored).
            
        Returns:
            Array of probability distributions.
        """
        if not self.is_fitted_:
            raise ValueError("Model not fitted. Call fit() first.")
        
        n_samples = X.shape[0]
        return np.tile(self.class_probabilities_, (n_samples, 1))

    def save(self, path: str | Path) -> None:
        """Save classifier state."""
        if not self.is_fitted_:
            raise ValueError("Model not fitted. Cannot save.")
        save_joblib(
            {
                "n_classes": self.n_classes,
                "most_frequent_class_": self.most_frequent_class_,
                "class_probabilities_": self.class_probabilities_,
                "is_fitted_": self.is_fitted_,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "MajorityClassifier":
        """Load classifier state."""
        data = load_joblib(path)
        model = cls(n_classes=data["n_classes"])
        model.most_frequent_class_ = data["most_frequent_class_"]
        model.class_probabilities_ = data["class_probabilities_"]
        model.is_fitted_ = data["is_fitted_"]
        return model


class MarkovClassifier:
    """Markov chain classifier for token sequences.
    
    Learns transition probabilities between tokens and predicts next token
    based on the last N tokens (n-gram model).
    """
    
    def __init__(self, order: int = 1, n_classes: int = 7):
        """Initialize Markov classifier.
        
        Args:
            order: Order of Markov chain (1 = bigram, 2 = trigram, etc.).
            n_classes: Number of possible token classes.
        """
        self.order = order
        self.n_classes = n_classes
        self.transition_matrix_: dict = None
        self.default_distribution_: np.ndarray = None
        self.is_fitted_ = False
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "MarkovClassifier":
        """Fit Markov classifier on token sequences.
        
        Args:
            X: Feature array with tokens (assumes tokens are in a column).
                 For sequence models, X should be 2D with shape (n_samples, window_size).
                 For tabular models, assumes last column contains the last token.
            y: Target labels (next token).
            
        Returns:
            Self.
        """
        contexts = self._contexts_from_X(X)
        
        # Filter invalid tokens
        valid_mask = y != -1
        valid_y = y[valid_mask]
        valid_contexts = [ctx for ctx, valid in zip(contexts, valid_mask) if valid and all(token != -1 for token in ctx)]
        
        if len(valid_y) == 0 or len(valid_contexts) == 0:
            raise ValueError("No valid samples found")
        
        # Build transition matrix
        self.transition_matrix_ = {}
        
        # Count transitions
        for context, next_token in zip(valid_contexts, valid_y):
            if 0 <= next_token < self.n_classes and all(0 <= token < self.n_classes for token in context):
                self.transition_matrix_.setdefault(context, np.zeros(self.n_classes))
                self.transition_matrix_[context][next_token] += 1
        
        # Normalize to probabilities
        for context, counts in self.transition_matrix_.items():
            total = counts.sum()
            if total > 0:
                self.transition_matrix_[context] = counts / total
        
        # Compute default distribution (overall class frequencies)
        unique_classes, counts = np.unique(valid_y, return_counts=True)
        self.default_distribution_ = np.zeros(self.n_classes)
        for cls, count in zip(unique_classes, counts):
            if 0 <= cls < self.n_classes:
                self.default_distribution_[cls] = count / len(valid_y)
        
        # Fill zeros in default distribution
        if self.default_distribution_.sum() == 0:
            self.default_distribution_ = np.ones(self.n_classes) / self.n_classes
        else:
            self.default_distribution_ /= self.default_distribution_.sum()
        
        self.is_fitted_ = True
        print(f"Markov classifier fitted: order={self.order}, n_classes={self.n_classes}")
        
        return self

    def _contexts_from_X(self, X: np.ndarray) -> list[tuple[int, ...]]:
        """Extract the Markov context from token windows."""
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        contexts = []
        for row in X:
            tokens = np.asarray(row).astype(int)
            if len(tokens) >= self.order:
                context = tuple(tokens[-self.order:])
            else:
                context = tuple(tokens)
            contexts.append(context)
        return contexts
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict next token using transition probabilities.
        
        Args:
            X: Feature array with tokens.
            
        Returns:
            Array of predicted class labels.
        """
        if not self.is_fitted_:
            raise ValueError("Model not fitted. Call fit() first.")
        
        contexts = self._contexts_from_X(X)
        
        predictions = []
        for context in contexts:
            if context in self.transition_matrix_:
                probs = self.transition_matrix_[context]
                pred = np.argmax(probs)
            else:
                # Use default distribution for unknown tokens
                pred = np.argmax(self.default_distribution_)
            predictions.append(pred)
        
        return np.array(predictions)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict transition probabilities.
        
        Args:
            X: Feature array with tokens.
            
        Returns:
            Array of probability distributions.
        """
        if not self.is_fitted_:
            raise ValueError("Model not fitted. Call fit() first.")
        
        contexts = self._contexts_from_X(X)
        
        probabilities = []
        for context in contexts:
            if context in self.transition_matrix_:
                probs = self.transition_matrix_[context]
            else:
                probs = self.default_distribution_
            probabilities.append(probs)
        
        return np.array(probabilities)

    def save(self, path: str | Path) -> None:
        """Save classifier state."""
        if not self.is_fitted_:
            raise ValueError("Model not fitted. Cannot save.")
        save_joblib(
            {
                "order": self.order,
                "n_classes": self.n_classes,
                "transition_matrix_": self.transition_matrix_,
                "default_distribution_": self.default_distribution_,
                "is_fitted_": self.is_fitted_,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "MarkovClassifier":
        """Load classifier state."""
        data = load_joblib(path)
        model = cls(order=data["order"], n_classes=data["n_classes"])
        model.transition_matrix_ = data["transition_matrix_"]
        model.default_distribution_ = data["default_distribution_"]
        model.is_fitted_ = data["is_fitted_"]
        return model


class LogisticRegressionBaseline:
    """Logistic regression baseline for tabular features.
    
    Optional baseline using sklearn's LogisticRegression.
    """
    
    def __init__(self, n_classes: int = 7, random_state: int = 42):
        """Initialize logistic regression baseline.
        
        Args:
            n_classes: Number of classes.
            random_state: Random seed for reproducibility.
        """
        self.n_classes = n_classes
        self.random_state = random_state
        self.model_ = None
        self.is_fitted_ = False
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionBaseline":
        """Fit logistic regression.
        
        Args:
            X: Feature array.
            y: Target labels.
            
        Returns:
            Self.
        """
        # Filter invalid labels
        valid_mask = y != -1
        X_valid = X[valid_mask]
        y_valid = y[valid_mask]
        
        if len(y_valid) == 0:
            raise ValueError("No valid labels found (all are -1)")
        
        # Rolling technical indicators intentionally leave early-window NaNs.
        # Tree models handle them natively; linear models need imputation.
        self.model_ = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(
                solver="lbfgs",
                max_iter=1000,
                random_state=self.random_state,
            ),
        )
        
        # Fit model
        self.model_.fit(X_valid, y_valid)
        self.is_fitted_ = True
        
        print(f"Logistic regression fitted: n_classes={self.n_classes}")
        print(f"Training accuracy: {self.model_.score(X_valid, y_valid):.4f}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.
        
        Args:
            X: Feature array.
            
        Returns:
            Array of predicted class labels.
        """
        if not self.is_fitted_:
            raise ValueError("Model not fitted. Call fit() first.")
        
        return self.model_.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.
        
        Args:
            X: Feature array.
            
        Returns:
            Array of probability distributions.
        """
        if not self.is_fitted_:
            raise ValueError("Model not fitted. Call fit() first.")

        proba = self.model_.predict_proba(X)
        estimator = self.model_.named_steps["logisticregression"]
        full_proba = np.zeros((len(X), self.n_classes))
        for source_idx, cls in enumerate(estimator.classes_):
            if 0 <= cls < self.n_classes:
                full_proba[:, cls] = proba[:, source_idx]
        return full_proba

    def save(self, path: str | Path) -> None:
        """Save classifier state."""
        if not self.is_fitted_:
            raise ValueError("Model not fitted. Cannot save.")
        save_joblib(
            {
                "n_classes": self.n_classes,
                "random_state": self.random_state,
                "model_": self.model_,
                "is_fitted_": self.is_fitted_,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "LogisticRegressionBaseline":
        """Load classifier state."""
        data = load_joblib(path)
        model = cls(n_classes=data["n_classes"], random_state=data["random_state"])
        model.model_ = data["model_"]
        model.is_fitted_ = data["is_fitted_"]
        return model


if __name__ == "__main__":
    # Example usage
    import numpy as np
    
    # Generate mock data
    np.random.seed(42)
    n_samples = 1000
    n_features = 50
    n_classes = 7
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, n_classes, n_samples)
    
    # Test majority classifier
    print("Testing MajorityClassifier:")
    majority = MajorityClassifier()
    majority.fit(X, y)
    pred = majority.predict(X[:10])
    print(f"Predictions: {pred}")
    
    # Test Markov classifier
    print("\nTesting MarkovClassifier:")
    # For Markov, X should contain tokens (last column)
    X_tokens = np.random.randint(0, n_classes, (n_samples, 1))
    markov = MarkovClassifier(order=1, n_classes=n_classes)
    markov.fit(X_tokens, y)
    pred = markov.predict(X_tokens[:10])
    print(f"Predictions: {pred}")
    
    # Test logistic regression
    print("\nTesting LogisticRegressionBaseline:")
    lr = LogisticRegressionBaseline(n_classes=n_classes, random_state=42)
    lr.fit(X, y)
    pred = lr.predict(X[:10])
    print(f"Predictions: {pred}")
