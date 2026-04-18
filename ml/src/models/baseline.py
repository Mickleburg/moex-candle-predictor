"""Baseline models for comparison."""

from typing import Optional

import numpy as np
from scipy.stats import mode
from sklearn.linear_model import LogisticRegression


class MajorityClassifier:
    """Majority class baseline.
    
    Always predicts the most frequent class from training data.
    """
    
    def __init__(self):
        """Initialize majority classifier."""
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
        n_classes = max(unique_classes) + 1
        self.class_probabilities_ = np.zeros(n_classes)
        for cls, count in zip(unique_classes, counts):
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
        # Extract last tokens from X
        if X.ndim == 2:
            # Assume last column or last position contains the last token
            if X.shape[1] > 1:
                last_tokens = X[:, -1].astype(int)
            else:
                last_tokens = X[:, 0].astype(int)
        else:
            last_tokens = X.astype(int)
        
        # Filter invalid tokens
        valid_mask = (y != -1) & (last_tokens != -1)
        valid_y = y[valid_mask]
        valid_last_tokens = last_tokens[valid_mask]
        
        if len(valid_y) == 0:
            raise ValueError("No valid samples found")
        
        # Build transition matrix
        self.transition_matrix_ = {}
        for i in range(self.n_classes):
            self.transition_matrix_[i] = np.zeros(self.n_classes)
        
        # Count transitions
        for prev_token, next_token in zip(valid_last_tokens, valid_y):
            if 0 <= prev_token < self.n_classes and 0 <= next_token < self.n_classes:
                self.transition_matrix_[prev_token][next_token] += 1
        
        # Normalize to probabilities
        for prev_token in range(self.n_classes):
            total = self.transition_matrix_[prev_token].sum()
            if total > 0:
                self.transition_matrix_[prev_token] /= total
            else:
                # Use uniform distribution if no transitions seen
                self.transition_matrix_[prev_token] = np.ones(self.n_classes) / self.n_classes
        
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
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict next token using transition probabilities.
        
        Args:
            X: Feature array with tokens.
            
        Returns:
            Array of predicted class labels.
        """
        if not self.is_fitted_:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Extract last tokens
        if X.ndim == 2:
            if X.shape[1] > 1:
                last_tokens = X[:, -1].astype(int)
            else:
                last_tokens = X[:, 0].astype(int)
        else:
            last_tokens = X.astype(int)
        
        predictions = []
        for token in last_tokens:
            if 0 <= token < self.n_classes:
                probs = self.transition_matrix_[token]
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
        
        # Extract last tokens
        if X.ndim == 2:
            if X.shape[1] > 1:
                last_tokens = X[:, -1].astype(int)
            else:
                last_tokens = X[:, 0].astype(int)
        else:
            last_tokens = X.astype(int)
        
        probabilities = []
        for token in last_tokens:
            if 0 <= token < self.n_classes:
                probs = self.transition_matrix_[token]
            else:
                probs = self.default_distribution_
            probabilities.append(probs)
        
        return np.array(probabilities)


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
        self.model_: LogisticRegression = None
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
        
        # Initialize model
        self.model_ = LogisticRegression(
            multi_class="multinomial",
            solver="lbfgs",
            max_iter=1000,
            random_state=self.random_state,
            n_jobs=-1
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
        
        return self.model_.predict_proba(X)


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
