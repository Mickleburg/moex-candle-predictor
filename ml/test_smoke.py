"""Smoke test for ML implementation."""

import sys
from pathlib import Path

# Add ml package root to path.
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "scripts"))


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


def test_nlp_alignment_invariants():
    """Test split/window/horizon accounting invariants."""

    print("\nTesting NLP alignment invariants...")

    try:
        import numpy as np

        from src.data.fixtures import generate_mock_candles
        from src.nlp import make_action_labels, make_sentence_samples, split_ranges
        from src.nlp.accounting import build_nlp_accounting_report
        from src.nlp.clustering import ClusterSpec

        df = generate_mock_candles(n=320, ticker="SBER", timeframe="1H", seed=42)
        horizon = 3
        window_size = 8
        ranges = split_ranges(len(df), train_ratio=0.7, val_ratio=0.15)
        labels, future_returns, _ = make_action_labels(df, horizon=horizon, commission=0.0005)
        word_tokens = [f"w{i % 6:03d}" for i in range(len(df))]

        for split_name, (split_start, split_end) in ranges.items():
            samples = make_sentence_samples(
                word_tokens,
                labels,
                future_returns,
                split_start,
                split_end,
                window_size,
                horizon,
            )
            split_len = split_end - split_start
            expected = split_len - window_size - horizon + 1
            assert samples.size == expected, (split_name, samples.size, expected)
            assert samples.target_indices[0] == split_start + window_size - 1
            assert samples.target_indices[-1] == split_end - horizon - 1
            assert np.all(samples.target_indices - window_size + 1 >= split_start)
            assert np.all(samples.target_indices + horizon < split_end)

        report = build_nlp_accounting_report(
            df,
            shape_variant="shape",
            horizon=horizon,
            window_size=window_size,
            cluster=ClusterSpec("kmeans", {"n_clusters": 6, "n_init": 3}),
        )
        assert all(report["checks"].values()), report["checks"]
        print("  PASS NLP split/window/horizon accounting")
        return True
    except Exception as exc:
        print(f"  FAIL NLP alignment test failed: {exc}")
        return False


def test_selection_uses_validation_only():
    """Test research best-selection helpers ignore test metrics."""

    print("\nTesting validation-only selection...")

    try:
        from sber_hourly_research import select_best_by_validation as select_hourly
        from sber_nlp_research import select_best_by_validation as select_nlp

        nlp_results = [
            {
                "status": "ok",
                "label": "first",
                "metrics": {"val": {"macro_f1": 0.4, "accuracy": 0.5}, "test": {"macro_f1": 0.9}},
            },
            {
                "status": "ok",
                "label": "second",
                "metrics": {"val": {"macro_f1": 0.4, "accuracy": 0.5}, "test": {"macro_f1": 0.1}},
            },
        ]
        assert select_nlp(nlp_results)["label"] == "first"

        hourly_results = [
            {"model_label": "first", "val": {"macro_f1": 0.4, "trade_action_accuracy": 0.5}, "test": {"macro_f1": 0.1}},
            {"model_label": "second", "val": {"macro_f1": 0.4, "trade_action_accuracy": 0.5}, "test": {"macro_f1": 0.9}},
        ]
        best, _ = select_hourly(hourly_results)
        assert best["model_label"] == "first"
        print("  PASS Selection ignores test metrics")
        return True
    except Exception as exc:
        print(f"  FAIL Selection test failed: {exc}")
        return False


def test_next_word_forecast_invariants():
    """Test next-word sample alignment and metrics."""

    print("\nTesting next-word forecast invariants...")

    try:
        import numpy as np

        from src.nlp.word_forecast import (
            PersistenceWordForecaster,
            evaluate_word_forecast,
            expected_next_word_sample_count,
            make_next_word_samples,
        )

        words = np.arange(80) % 5
        split_start, split_end = 10, 60
        context_size, forecast_horizon = 6, 4
        samples = make_next_word_samples(words, split_start, split_end, context_size, forecast_horizon)
        expected = expected_next_word_sample_count(split_end - split_start, context_size, forecast_horizon)
        assert samples.size == expected
        assert samples.sample_indices[0] == split_start + context_size - 1
        assert samples.sample_indices[-1] == split_end - forecast_horizon - 1
        assert np.all(samples.X_contexts[:, -1] == words[samples.sample_indices])
        for row_idx, t_idx in enumerate(samples.sample_indices):
            assert np.array_equal(samples.Y_future_words[row_idx], words[t_idx + 1 : t_idx + forecast_horizon + 1])

        model = PersistenceWordForecaster().fit(samples.X_contexts, samples.Y_future_words, n_words=5)
        pred = model.predict(samples.X_contexts)
        distances = np.abs(np.subtract.outer(np.arange(5), np.arange(5))).astype(float)
        metrics = evaluate_word_forecast(samples.Y_future_words, pred, n_words=5, distance_matrix=distances)
        assert "mean_soft_similarity" in metrics
        assert len(metrics["per_horizon"]) == forecast_horizon
        print("  PASS Next-word samples and metrics")
        return True
    except Exception as exc:
        print(f"  FAIL Next-word invariant test failed: {exc}")
        return False


def test_walk_forward_invariants():
    """Test walk-forward fold ordering and train-only Markov priors."""

    print("\nTesting walk-forward invariants...")

    try:
        import numpy as np

        from src.data.split import walk_forward_ranges
        from src.nlp.word_forecast import (
            expected_next_word_sample_count,
            fit_markov_prior_features,
            make_markov_prior_feature_matrix,
            make_next_word_samples,
        )

        folds = walk_forward_ranges(
            100,
            n_splits=3,
            initial_train_size=40,
            val_size=15,
            gap=0,
            min_train_size=20,
        )
        assert len(folds) == 3
        previous_train_end = 0
        words = np.arange(100) % 5
        context_size, forecast_horizon = 6, 3
        for fold in folds:
            assert fold.train_start == 0
            assert fold.train_end <= fold.val_start
            assert fold.val_end <= 100
            assert fold.train_end > previous_train_end
            previous_train_end = fold.train_end
            train_samples = make_next_word_samples(
                words,
                fold.train_start,
                fold.train_end,
                context_size,
                forecast_horizon,
            )
            val_samples = make_next_word_samples(
                words,
                fold.val_start,
                fold.val_end,
                context_size,
                forecast_horizon,
            )
            assert train_samples.size == expected_next_word_sample_count(fold.train_len, context_size, forecast_horizon)
            assert val_samples.size == expected_next_word_sample_count(fold.val_len, context_size, forecast_horizon)
            assert np.all(val_samples.sample_indices - context_size + 1 >= fold.val_start)
            assert np.all(val_samples.sample_indices + forecast_horizon < fold.val_end)

        train_only_words = np.array([0, 1, 0, 1, 0, 2, 2, 2, 2, 2])
        prior = fit_markov_prior_features(train_only_words, train_start=0, train_end=5, n_words=3)
        assert prior.transition_matrix[0, 1] == 1.0
        assert prior.transition_matrix[0, 2] == 0.0
        features = make_markov_prior_feature_matrix(train_only_words, [4, 5], prior)
        assert features.shape == (2, 6)
        assert features[0, 1] == 1.0
        print("  PASS Walk-forward ranges and train-only Markov priors")
        return True
    except Exception as exc:
        print(f"  FAIL Walk-forward invariant test failed: {exc}")
        return False


def test_word_lm_invariants():
    """Test n-gram LM probabilities, decoding, and train-only fit."""

    print("\nTesting word language-model invariants...")

    try:
        import numpy as np

        from src.nlp.word_forecast import make_next_word_samples
        from src.nlp.word_lm import NGramBackoffLanguageModel, evaluate_language_model

        words = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 2, 0, 2, 0, 2])
        train_start, train_end = 0, 8
        val_start, val_end = 8, len(words)
        model = NGramBackoffLanguageModel(order=2, alpha=0.1).fit(
            words,
            train_start=train_start,
            train_end=train_end,
            n_words=3,
        )
        proba = model.next_proba([0])
        assert np.isclose(proba.sum(), 1.0)
        assert proba[1] > proba[2], "validation-only transition leaked into train counts"

        samples = make_next_word_samples(words, val_start, val_end, context_size=2, forecast_horizon=3)
        decoded = model.greedy_decode(samples.X_contexts[0], forecast_horizon=3)
        assert len(decoded) == 3
        beam = model.beam_search(samples.X_contexts[0], forecast_horizon=3, beam_width=3)
        assert len(beam) == 3
        metrics = evaluate_language_model(model, samples.X_contexts, samples.Y_future_words, beam_width=3)
        assert np.isfinite(metrics["mean_token_nll"])
        assert np.isfinite(metrics["perplexity"])
        assert metrics["perplexity"] > 0
        assert "beam_contains_true_sequence" in metrics
        print("  PASS Word LM probabilities, NLL, perplexity, and train-only counts")
        return True
    except Exception as exc:
        print(f"  FAIL Word LM invariant test failed: {exc}")
        return False


def test_predictor_input_validation():
    """Test inference preprocessing sorts and rejects ambiguous inputs."""

    print("\nTesting predictor input validation...")

    try:
        from datetime import datetime, timedelta

        from src.service import CandlePredictor

        predictor = CandlePredictor()
        base = datetime(2024, 1, 1, 10)
        candles = [
            {"begin": base + timedelta(hours=2), "open": 1, "high": 2, "low": 1, "close": 1.5, "volume": 10, "ticker": "SBER", "timeframe": "1H"},
            {"begin": base, "open": 1, "high": 2, "low": 1, "close": 1.5, "volume": 10, "ticker": "SBER", "timeframe": "1H"},
            {"begin": base + timedelta(hours=1), "open": 1, "high": 2, "low": 1, "close": 1.5, "volume": 10, "ticker": "SBER", "timeframe": "1H"},
        ]
        df = predictor._candles_to_dataframe(candles)
        assert df["begin"].is_monotonic_increasing

        duplicate = [candles[0], candles[0]]
        try:
            predictor._candles_to_dataframe(duplicate)
            raise AssertionError("duplicate begin was accepted")
        except ValueError:
            pass

        mixed = [dict(candles[0]), dict(candles[1], ticker="GAZP")]
        try:
            predictor._candles_to_dataframe(mixed)
            raise AssertionError("mixed ticker was accepted")
        except ValueError:
            pass

        mixed_timeframe = [dict(candles[0]), dict(candles[1], timeframe="10min")]
        try:
            predictor._candles_to_dataframe(mixed_timeframe)
            raise AssertionError("mixed timeframe was accepted")
        except ValueError:
            pass

        print("  PASS Predictor input validation")
        return True
    except Exception as exc:
        print(f"  FAIL Predictor validation test failed: {exc}")
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
        ("NLP Alignment", test_nlp_alignment_invariants()),
        ("Validation Selection", test_selection_uses_validation_only()),
        ("Next-word Forecast", test_next_word_forecast_invariants()),
        ("Walk-forward Invariants", test_walk_forward_invariants()),
        ("Word LM Invariants", test_word_lm_invariants()),
        ("Predictor Validation", test_predictor_input_validation()),
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
