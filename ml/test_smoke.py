"""Smoke test for ML implementation."""

import sys
from pathlib import Path

# Add ml package root to path.
sys.path.insert(0, str(Path(__file__).parent))


def test_imports():
    """Test that all modules can be imported."""

    print("Testing imports...")

    try:
        from src.data import clean_candles, load_candles, time_split

        print("  PASS Data modules imported")
    except Exception as exc:
        print(f"  FAIL Data modules failed: {exc}")
        return False

    try:
        from src.features import CandleTokenizer, build_tabular_windows, compute_all_indicators

        print("  PASS Features modules imported")
    except Exception as exc:
        print(f"  FAIL Features modules failed: {exc}")
        return False

    try:
        from src.models import LGBMClassifier, MajorityClassifier

        print("  PASS Models modules imported")
    except Exception as exc:
        print(f"  FAIL Models modules failed: {exc}")
        return False

    try:
        from src.evaluation import compute_classification_metrics

        print("  PASS Evaluation modules imported")
    except Exception as exc:
        print(f"  FAIL Evaluation modules failed: {exc}")
        return False

    try:
        from src.nlp import CandleClusterer, ClassifierSpec, ClusterSpec, VectorizerSpec

        print("  PASS NLP candle-language modules imported")
    except Exception as exc:
        print(f"  FAIL NLP candle-language modules failed: {exc}")
        return False

    try:
        from src.service import CandlePredictor

        print("  PASS Service modules imported")
    except Exception as exc:
        print(f"  FAIL Service modules failed: {exc}")
        return False

    return True


def test_config_loading():
    """Test that configs can be loaded."""

    print("\nTesting config loading...")

    try:
        from src.utils.config import load_all_configs

        configs = load_all_configs(Path(__file__).parent / "configs")
        print(f"  PASS Loaded {len(configs)} configs")
        return True
    except Exception as exc:
        print(f"  FAIL Config loading failed: {exc}")
        return False


def test_mock_pipeline():
    """Test pipeline with mock data."""

    print("\nTesting pipeline with mock data...")

    try:
        from src.data.fixtures import generate_mock_candles
        from src.features import CandleTokenizer, compute_all_indicators

        df = generate_mock_candles(n=100, ticker="SBER", timeframe="1H", seed=42)
        print(f"  PASS Generated mock data: {len(df)} candles")

        features_df = compute_all_indicators(df)
        print(f"  PASS Computed features: {features_df.shape}")

        tokenizer = CandleTokenizer(n_bins=7, horizon=3, random_state=42)
        tokens = tokenizer.fit_transform(features_df)
        print(f"  PASS Tokenized data: {tokens.shape}")

        return True
    except Exception as exc:
        print(f"  FAIL Pipeline test failed: {exc}")
        return False


def test_nlp_pipeline():
    """Test the candle-language pipeline with mock data."""

    print("\nTesting NLP candle-language pipeline...")

    try:
        from src.data.fixtures import generate_mock_candles
        from src.nlp import ClassifierSpec, ClusterSpec, ExperimentConfig, VectorizerSpec, run_experiment

        df = generate_mock_candles(n=320, ticker="SBER", timeframe="1H", seed=42)
        config = ExperimentConfig(
            shape_variant="ohlc",
            horizon=1,
            window_size=8,
            commission=0.0005,
            cluster=ClusterSpec("kmeans", {"n_clusters": 6, "n_init": 3}),
            vectorizer=VectorizerSpec("tfidf", {"ngram_range": (1, 2), "min_df": 1}),
            classifier=ClassifierSpec("ridge", {"alpha": 1.0}),
        )
        result = run_experiment(df, config, random_state=42)
        print(f"  PASS NLP val macro_f1: {result['metrics']['val']['macro_f1']:.4f}")
        return True
    except Exception as exc:
        print(f"  FAIL NLP pipeline test failed: {exc}")
        return False


def main():
    """Run all smoke tests."""

    print("=" * 50)
    print("ML Implementation Smoke Tests")
    print("=" * 50)

    results = [
        ("Imports", test_imports()),
        ("Config Loading", test_config_loading()),
        ("Mock Pipeline", test_mock_pipeline()),
        ("NLP Pipeline", test_nlp_pipeline()),
    ]

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

    print("Some tests failed!")
    return 1


if __name__ == "__main__":
    sys.exit(main())
