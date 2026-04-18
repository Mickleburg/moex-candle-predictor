"""RNN model stub for sequence data.

TODO: This is a lightweight stub for future implementation.
The main MVP uses LightGBM (tabular model). RNN implementation is deferred
to avoid adding heavy dependencies (PyTorch/TensorFlow) and keep the
dependency graph simple for the CPU-only MVP.

Future implementation should:
- Add PyTorch or TensorFlow dependency to requirements.txt
- Implement embedding layer for token IDs
- Implement GRU/LSTM layer for sequence modeling
- Implement output layer for multiclass classification
- Support CPU training (no GPU required)
- Match the same API as other models (fit, predict, predict_proba, save, load)
"""

from pathlib import Path
from typing import Optional

import numpy as np


class RNNClassifier:
    """RNN classifier stub for sequence data.
    
    TODO: Implement full RNN model with embedding + GRU/LSTM.
    Current implementation is a stub that provides the interface
    but raises NotImplementedError for actual training/inference.
    """
    
    def __init__(
        self,
        n_classes: int = 7,
        embedding_dim: int = 32,
        hidden_dim: int = 64,
        num_layers: int = 1,
        random_state: int = 42
    ):
        """Initialize RNN classifier stub.
        
        Args:
            n_classes: Number of output classes.
            embedding_dim: Dimension of token embedding.
            hidden_dim: Hidden state dimension of RNN.
            num_layers: Number of RNN layers.
            random_state: Random seed for reproducibility.
        """
        self.n_classes = n_classes
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.random_state = random_state
        
        self.model_ = None
        self.is_fitted_ = False
        
        print("WARNING: RNNClassifier is a stub. Full implementation requires PyTorch/TensorFlow.")
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> "RNNClassifier":
        """Fit RNN classifier (stub).
        
        TODO: Implement full training loop with:
        - Embedding layer for token IDs
        - GRU/LSTM layer for sequence processing
        - Linear output layer for classification
        - Cross-entropy loss
        - Adam optimizer
        - Early stopping on validation set
        
        Args:
            X: Sequence feature array shape (n_samples, seq_len, n_features).
            y: Target labels.
            X_val: Optional validation sequences.
            y_val: Optional validation labels.
            
        Returns:
            Self.
            
        Raises:
            NotImplementedError: This is a stub.
        """
        raise NotImplementedError(
            "RNNClassifier is a stub. To implement:\n"
            "1. Add PyTorch or TensorFlow to requirements.txt\n"
            "2. Implement embedding layer for token IDs\n"
            "3. Implement GRU/LSTM layer for sequence modeling\n"
            "4. Implement output layer for multiclass classification\n"
            "5. Add training loop with loss and optimizer\n"
            "6. Support CPU-only training"
        )
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels (stub).
        
        TODO: Implement forward pass through trained RNN.
        
        Args:
            X: Sequence feature array shape (n_samples, seq_len, n_features).
            
        Returns:
            Array of predicted class labels.
            
        Raises:
            NotImplementedError: This is a stub.
        """
        raise NotImplementedError(
            "RNNClassifier is a stub. Implement predict() method after model training."
        )
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities (stub).
        
        TODO: Implement forward pass with softmax output.
        
        Args:
            X: Sequence feature array shape (n_samples, seq_len, n_features).
            
        Returns:
            Array of probability distributions.
            
        Raises:
            NotImplementedError: This is a stub.
        """
        raise NotImplementedError(
            "RNNClassifier is a stub. Implement predict_proba() method after model training."
        )
    
    def save(self, path: str | Path) -> None:
        """Save model to file (stub).
        
        TODO: Implement model serialization using torch.save or tf.keras.models.save_model.
        
        Args:
            path: Path to save model.
            
        Raises:
            NotImplementedError: This is a stub.
        """
        raise NotImplementedError(
            "RNNClassifier is a stub. Implement save() method using framework-specific serialization."
        )
    
    @classmethod
    def load(cls, path: str | Path) -> "RNNClassifier":
        """Load model from file (stub).
        
        TODO: Implement model deserialization using torch.load or tf.keras.models.load_model.
        
        Args:
            path: Path to load model from.
            
        Returns:
            Loaded model instance.
            
        Raises:
            NotImplementedError: This is a stub.
        """
        raise NotImplementedError(
            "RNNClassifier is a stub. Implement load() method using framework-specific deserialization."
        )


if __name__ == "__main__":
    # Example usage (will raise NotImplementedError)
    import numpy as np
    
    print("Testing RNNClassifier stub:")
    
    rnn = RNNClassifier(
        n_classes=7,
        embedding_dim=32,
        hidden_dim=64,
        random_state=42
    )
    
    # Generate mock sequence data
    n_samples = 100
    seq_len = 32
    n_features = 20
    
    X = np.random.randn(n_samples, seq_len, n_features)
    y = np.random.randint(0, 7, n_samples)
    
    print(f"Input shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    try:
        rnn.fit(X, y)
    except NotImplementedError as e:
        print(f"\nExpected error (stub): {e}")
