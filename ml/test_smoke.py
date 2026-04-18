"""Smoke test for ML implementation."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from src.data import load_candles, clean_candles, time_split
        print("  ✓ Data modules imported")
    except Exception as e:
        print(f"  ✗ Data modules failed: {e}")
        return False
    
    try:
        from src.features import compute_all_indicators, CandleTokenizer, build_tabular_windows
        print("  ✓ Features modules imported")
    except Exception as e:
        print(f"  ✗ Features modules failed: {e}")
        return False
    
    try:
        from src.models import LGBMClassifier, MajorityClassifier
        print("  ✓ Models modules imported")
    except Exception as e:
        print(f"  ✗ Models modules failed: {e}")
        return False
    
    try:
        from src.evaluation import compute_classification_metrics
        print("  ✓ Evaluation modules imported")
    except Exception as e:
        print(f"  ✗ Evaluation modules failed: {e}")
        return False
    
    try:
        from src.service import CandlePredictor
        print("  ✓ Service modules imported")
    except Exception as e:
        print(f"  ✗ Service modules failed: {e}")
        return False
    
    return True


def test_config_loading():
    """Test that configs can be loaded."""
    print("\nTesting config loading...")
    
    try:
        from src.utils.config import load_all_configs
        configs = load_all_configs("configs")
        print(f"  ✓ Loaded {len(configs)} configs")
        return True
    except Exception as e:
        print(f"  ✗ Config loading failed: {e}")
        return False


def test_mock_pipeline():
    """Test pipeline with mock data."""
    print("\nTesting pipeline with mock data...")
    
    try:
        from src.data.fixtures import generate_mock_candles
        from src.features import compute_all_indicators, CandleTokenizer
        
        # Generate mock data
        df = generate_mock_candles(n=100, ticker="SBER", timeframe="1H", seed=42)
        print(f"  ✓ Generated mock data: {len(df)} candles")
        
        # Compute features
        features_df = compute_all_indicators(df)
        print(f"  ✓ Computed features: {features_df.shape}")
        
        # Fit tokenizer
        tokenizer = CandleTokenizer(n_bins=7, horizon=3, random_state=42)
        tokens = tokenizer.fit_transform(features_df)
        print(f"  ✓ Tokenized data: {tokens.shape}")
        
        return True
    except Exception as e:
        print(f"  ✗ Pipeline test failed: {e}")
        return False


def main():
    """Run all smoke tests."""
    print("=" * 50)
    print("ML Implementation Smoke Tests")
    print("=" * 50)
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("Config Loading", test_config_loading()))
    results.append(("Mock Pipeline", test_mock_pipeline()))
    
    print("\n" + "=" * 50)
    print("Results Summary")
    print("=" * 50)
    
    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 50)
    
    if all_passed:
        print("All tests passed!")
        return 0
    else:
        print("Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
