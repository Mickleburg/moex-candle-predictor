"""LightGBM multiclass classifier for tabular data."""

from pathlib import Path
from typing import Optional

import numpy as np

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

from ..utils.io import load_joblib, save_joblib


class LGBMClassifier:
    """LightGBM multiclass classifier wrapper.
    
    Main MVP model for tabular features.
    Uses CPU-friendly hyperparameters for small datasets.
    """
    
    def __init__(
        self,
        n_classes: int = 7,
        random_state: int = 42,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        num_leaves: int = 31,
        min_child_samples: int = 20,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.1,
        reg_lambda: float = 0.1,
        verbose: int = -1
    ):
        """Initialize LightGBM classifier.
        
        Args:
            n_classes: Number of classes.
            random_state: Random seed for reproducibility.
            n_estimators: Number of boosting rounds.
            max_depth: Maximum tree depth.
            learning_rate: Learning rate.
            num_leaves: Maximum number of leaves in one tree.
            min_child_samples: Minimum number of data in one leaf.
            subsample: Subsample ratio of training instances.
            colsample_bytree: Subsample ratio of columns when constructing each tree.
            reg_alpha: L1 regularization.
            reg_lambda: L2 regularization.
            verbose: Verbosity level (-1 for silent).
        """
        if not LIGHTGBM_AVAILABLE:
            raise ImportError(
                "LightGBM is not installed. Install it with: pip install lightgbm"
            )
        
        self.n_classes = n_classes
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.verbose = verbose
        
        self.model_: lgb.LGBMClassifier = None
        self.is_fitted_ = False
        
        # Class imbalance: use balanced class weights
        # This is a simple approach that works well for MVP
        self.class_weight: str = "balanced"
    
    def _create_model(self) -> lgb.LGBMClassifier:
        """Create LightGBM model with configured parameters."""
        model = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=self.n_classes,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            min_child_samples=self.min_child_samples,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            class_weight=self.class_weight,
            random_state=self.random_state,
            verbose=self.verbose,
            n_jobs=-1,  # Use all CPU cores
            force_col_wise=True,  # CPU optimization
        )
        return model
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> "LGBMClassifier":
        """Fit LightGBM classifier.
        
        Args:
            X: Training feature array.
            y: Training labels.
            X_val: Optional validation feature array.
            y_val: Optional validation labels.
            
        Returns:
            Self.
        """
        # Filter invalid labels
        valid_mask = y != -1
        X_valid = X[valid_mask]
        y_valid = y[valid_mask]
        
        if len(y_valid) == 0:
            raise ValueError("No valid labels found (all are -1)")
        
        # Create model
        self.model_ = self._create_model()
        
        # Fit with optional validation set
        if X_val is not None and y_val is not None:
            # Filter validation set
            val_mask = y_val != -1
            X_val_filtered = X_val[val_mask]
            y_val_filtered = y_val[val_mask]
            
            if len(y_val_filtered) > 0:
                self.model_.fit(
                    X_valid,
                    y_valid,
                    eval_set=[(X_val_filtered, y_val_filtered)],
                    callbacks=[
                        lgb.early_stopping(stopping_rounds=10, verbose=False),
                        lgb.log_evaluation(period=0)
                    ]
                )
            else:
                self.model_.fit(X_valid, y_valid)
        else:
            self.model_.fit(X_valid, y_valid)
        
        self.is_fitted_ = True
        
        # Print training info
        train_score = self.model_.score(X_valid, y_valid)
        print(f"LightGBM fitted: n_classes={self.n_classes}")
        print(f"Training accuracy: {train_score:.4f}")
        
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
    
    def get_feature_importance(self, importance_type: str = "gain") -> np.ndarray:
        """Get feature importance scores.
        
        Args:
            importance_type: Type of importance ('gain', 'split', etc.).
            
        Returns:
            Array of feature importance scores.
        """
        if not self.is_fitted_:
            raise ValueError("Model not fitted. Call fit() first.")

        booster = self.model_.booster_
        return booster.feature_importance(importance_type=importance_type)
    
    def save(self, path: str | Path) -> None:
        """Save model to file.
        
        Args:
            path: Path to save model.
        """
        if not self.is_fitted_:
            raise ValueError("Model not fitted. Cannot save.")
        
        path = Path(path)
        
        save_data = {
            "model": self.model_,
            "n_classes": self.n_classes,
            "random_state": self.random_state,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "min_child_samples": self.min_child_samples,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "class_weight": self.class_weight,
            "is_fitted": self.is_fitted_,
        }
        
        save_joblib(save_data, path)
        print(f"LightGBM model saved to {path}")
    
    @classmethod
    def load(cls, path: str | Path) -> "LGBMClassifier":
        """Load model from file.
        
        Args:
            path: Path to load model from.
            
        Returns:
            Loaded model instance.
        """
        path = Path(path)
        
        save_data = load_joblib(path)
        
        # Reconstruct classifier
        classifier = cls(
            n_classes=save_data["n_classes"],
            random_state=save_data["random_state"],
            n_estimators=save_data["n_estimators"],
            max_depth=save_data["max_depth"],
            learning_rate=save_data["learning_rate"],
            num_leaves=save_data["num_leaves"],
            min_child_samples=save_data["min_child_samples"],
            subsample=save_data["subsample"],
            colsample_bytree=save_data["colsample_bytree"],
            reg_alpha=save_data["reg_alpha"],
            reg_lambda=save_data["reg_lambda"],
        )
        
        classifier.model_ = save_data["model"]
        classifier.class_weight = save_data["class_weight"]
        classifier.is_fitted_ = save_data["is_fitted"]
        
        print(f"LightGBM model loaded from {path}")
        
        return classifier


if __name__ == "__main__":
    # Example usage
    import numpy as np
    
    # Generate mock data
    np.random.seed(42)
    n_samples = 1000
    n_features = 50
    n_classes = 7
    
    X_train = np.random.randn(n_samples, n_features)
    y_train = np.random.randint(0, n_classes, n_samples)
    
    X_val = np.random.randn(200, n_features)
    y_val = np.random.randint(0, n_classes, 200)
    
    # Test LightGBM classifier
    print("Testing LGBMClassifier:")
    lgbm = LGBMClassifier(
        n_classes=n_classes,
        random_state=42,
        n_estimators=50,  # Small for quick test
        verbose=-1
    )
    lgbm.fit(X_train, y_train, X_val, y_val)
    
    pred = lgbm.predict(X_val[:10])
    print(f"Predictions: {pred}")
    
    proba = lgbm.predict_proba(X_val[:10])
    print(f"Probabilities shape: {proba.shape}")
    
    # Feature importance
    importance = lgbm.get_feature_importance(importance_type="gain")
    print(f"Feature importance shape: {importance.shape}")
