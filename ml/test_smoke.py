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

        from src.data.split import rolling_walk_forward_ranges, walk_forward_ranges
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

        rolling = rolling_walk_forward_ranges(100, train_size=40, val_size=15, step_size=10, max_folds=3)
        assert [(fold.train_start, fold.train_end, fold.val_start, fold.val_end) for fold in rolling] == [
            (0, 40, 40, 55),
            (10, 50, 50, 65),
            (20, 60, 60, 75),
        ]
        for fold in rolling:
            assert fold.train_len == 40
            assert fold.val_len == 15
            assert fold.train_end <= fold.val_start
        print("  PASS Walk-forward ranges, rolling folds, and train-only Markov priors")
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
        from src.nlp.word_lm import NGramBackoffLanguageModel, confidence_analysis, evaluate_language_model

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
        confidence = confidence_analysis(model, samples.X_contexts, samples.Y_future_words, thresholds=(0.99,))
        assert confidence["confidence_buckets"]
        assert confidence["abstention_curves"]["top1_probability"][0]["coverage"] >= 0.0
        print("  PASS Word LM probabilities, NLL, perplexity, and train-only counts")
        return True
    except Exception as exc:
        print(f"  FAIL Word LM invariant test failed: {exc}")
        return False


def test_lm_action_feature_invariants():
    """Test LM-derived action features are aligned and finite."""

    print("\nTesting LM action feature invariants...")

    try:
        import inspect
        import numpy as np

        from src.nlp.action_features import make_lm_action_features
        from src.nlp.word_forecast import make_next_word_samples
        from src.nlp.word_lm import NGramBackoffLanguageModel

        words = np.arange(80) % 6
        split_start, split_end = 20, 70
        context_size = 8
        samples = make_next_word_samples(words, split_start, split_end, context_size=context_size, forecast_horizon=1)
        model = NGramBackoffLanguageModel(order=2, alpha=0.1).fit(words, train_start=0, train_end=split_start, n_words=6)
        distances = np.abs(np.subtract.outer(np.arange(6), np.arange(6))).astype(float)
        features = make_lm_action_features(
            word_ids=words,
            target_indices=samples.sample_indices,
            context_size=context_size,
            model=model,
            distance_matrix=distances,
            include_probabilities=True,
        )
        assert features.X.shape[0] == samples.size
        assert features.X.shape[1] == len(features.names)
        assert np.all(np.isfinite(features.X))
        signature = inspect.signature(make_lm_action_features)
        assert "Y_future_words" not in signature.parameters
        assert "future_words" not in signature.parameters
        print("  PASS LM action features aligned, finite, and target-free")
        return True
    except Exception as exc:
        print(f"  FAIL LM action feature invariant test failed: {exc}")
        return False


def test_action_lm_robustness_invariants():
    """Test action LM robustness helpers stay leakage-safe on mock data."""

    print("\nTesting action LM robustness invariants...")

    try:
        import numpy as np

        from sber_action_lm_features_walk_forward import (
            calibration_diagnostics,
            parse_int_list,
            regime_error_analysis,
            threshold_sweep,
        )
        from src.data.fixtures import generate_mock_candles
        from src.data.split import walk_forward_ranges
        from src.nlp import make_action_labels, make_sentence_samples

        assert parse_int_list("7,13,42") == [7, 13, 42]
        folds_a = walk_forward_ranges(120, n_splits=2, initial_train_size=60, val_size=20, min_train_size=40)
        folds_b = walk_forward_ranges(120, n_splits=2, initial_train_size=60, val_size=20, min_train_size=40)
        assert [fold.__dict__ for fold in folds_a] == [fold.__dict__ for fold in folds_b]

        y_true = np.array([0, 1, 2, 1, 2, 0])
        y_pred = np.array([0, 1, 1, 1, 2, 2])
        proba = np.array(
            [
                [0.70, 0.20, 0.10],
                [0.10, 0.75, 0.15],
                [0.20, 0.45, 0.35],
                [0.15, 0.60, 0.25],
                [0.10, 0.20, 0.70],
                [0.40, 0.20, 0.40],
            ],
            dtype=float,
        )
        assert np.allclose(proba.sum(axis=1), 1.0)
        calibration = calibration_diagnostics(y_true, y_pred, proba)
        assert calibration["available"]
        assert len(calibration["reliability_table"]) == 6
        sweep = threshold_sweep(y_true, proba)
        assert sweep["available"]
        assert len(sweep["rows"]) == 25

        df = generate_mock_candles(n=140, ticker="SBER", timeframe="1H", seed=42)
        labels, future_returns, _ = make_action_labels(df, horizon=1, commission=0.0005)
        word_tokens = [f"w{i % 5:03d}" for i in range(len(df))]
        train_samples = make_sentence_samples(word_tokens, labels, future_returns, 0, 90, 8, 1)
        val_samples = make_sentence_samples(word_tokens, labels, future_returns, 90, 130, 8, 1)
        fake_val_pred = val_samples.y.copy()
        fake_proba = np.full((val_samples.size, 3), 1.0 / 3.0)
        train_lm_scalar = np.zeros((train_samples.size, 18), dtype=float)
        val_lm_scalar = np.zeros((val_samples.size, 18), dtype=float)
        train_lm_scalar[:, 3] = np.linspace(0.1, 1.0, train_samples.size)
        val_lm_scalar[:, 3] = np.linspace(0.1, 1.0, val_samples.size)
        val_lm_scalar[:, 0] = 0.4
        regimes = regime_error_analysis(
            df,
            train_samples,
            val_samples,
            fake_val_pred,
            fake_proba,
            train_lm_scalar,
            val_lm_scalar,
        )
        assert {"volatility", "trend", "session", "lm_uncertainty"} <= set(regimes)
        for rows in regimes.values():
            assert sum(row["n_samples"] for row in rows) == val_samples.size
        print("  PASS Action LM robustness helpers")
        return True
    except Exception as exc:
        print(f"  FAIL Action LM robustness invariant test failed: {exc}")
        return False


def test_nested_threshold_invariants():
    """Test nested threshold selection does not use outer validation for honest modes."""

    print("\nTesting nested threshold invariants...")

    try:
        import numpy as np

        from sber_action_nested_thresholds import (
            apply_temperature,
            nested_range,
            select_global_thresholds,
            select_regime_thresholds,
            threshold_predictions,
        )
        from src.data.split import WalkForwardRange

        fold = WalkForwardRange(fold_id=1, train_start=0, train_end=80, val_start=80, val_end=110)
        ranges = nested_range(fold, calibration_size=20)
        assert ranges.inner_train_start == 0
        assert ranges.inner_train_end == 60
        assert ranges.calibration_start == 60
        assert ranges.calibration_end == 80
        assert ranges.calibration_end <= ranges.outer_fold.val_start

        proba = np.array(
            [
                [0.60, 0.25, 0.15],
                [0.20, 0.55, 0.25],
                [0.15, 0.30, 0.55],
                [0.35, 0.30, 0.35],
            ],
            dtype=float,
        )
        calibrated = apply_temperature(proba, 1.25)
        assert np.all(np.isfinite(calibrated))
        assert np.allclose(calibrated.sum(axis=1), 1.0)
        pred = threshold_predictions(calibrated, buy_threshold=0.3, sell_threshold=0.3)
        assert pred.shape == (4,)

        y = np.array([0, 1, 2, 1])
        grid = [0.25, 0.30, 0.35]
        temps = [0.75, 1.0]
        decision = select_global_thresholds(y, proba, grid, temps, selection_metric="macro_f1", mode="global")
        assert not decision.is_oracle
        assert decision.buy_threshold in grid
        assert decision.sell_threshold in grid
        assert decision.temperature in temps

        regimes = np.array(["a", "a", "b", "b"], dtype=object)
        regime_decision = select_regime_thresholds(
            y,
            proba,
            regimes,
            grid,
            temps,
            selection_metric="macro_f1",
            mode="regime_test",
            min_regime_calibration_samples=3,
        )
        assert regime_decision.regime_thresholds is not None
        assert regime_decision.regime_thresholds["a"]["fallback"]
        assert regime_decision.regime_thresholds["b"]["fallback"]

        oracle = select_global_thresholds(y, proba, grid, temps, selection_metric="macro_f1", mode="oracle_global", is_oracle=True)
        assert oracle.is_oracle
        print("  PASS Nested threshold selection helpers")
        return True
    except Exception as exc:
        print(f"  FAIL Nested threshold invariant test failed: {exc}")
        return False


def test_vocabulary_selection_constraints():
    """Test vocabulary selection constraints and rejection reasons."""

    print("\nTesting vocabulary selection constraints...")

    try:
        from sber_word_lm_walk_forward import apply_vocabulary_constraints

        rows = [
            {
                "shape_variant": "shape",
                "clusterer": "gmm_diag",
                "vocab_size_requested": 20,
                "normalized_entropy_mean": 0.7,
                "dominant_share_mean": 0.3,
                "top3_share_mean": 0.7,
                "observed_vocab_ratio_mean": 1.0,
            },
            {
                "shape_variant": "ohlc",
                "clusterer": "kmeans",
                "vocab_size_requested": 8,
                "normalized_entropy_mean": 0.3,
                "dominant_share_mean": 0.8,
                "top3_share_mean": 0.95,
                "observed_vocab_ratio_mean": 1.0,
            },
        ]
        constrained = apply_vocabulary_constraints(
            rows,
            min_norm_entropy=0.5,
            max_dominant_share=0.55,
            max_top3_share=0.8,
            min_observed_vocab_ratio=0.8,
        )
        assert constrained[0]["accepted_by_constraints"]
        assert constrained[0]["rejection_reason"] == ""
        assert not constrained[1]["accepted_by_constraints"]
        assert "normalized_entropy" in constrained[1]["rejection_reason"]
        assert "dominant_share" in constrained[1]["rejection_reason"]
        assert "top3_share" in constrained[1]["rejection_reason"]
        print("  PASS Vocabulary constraints and rejection reasons")
        return True
    except Exception as exc:
        print(f"  FAIL Vocabulary selection constraint test failed: {exc}")
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
        ("LM Action Features", test_lm_action_feature_invariants()),
        ("Action LM Robustness", test_action_lm_robustness_invariants()),
        ("Nested Thresholds", test_nested_threshold_invariants()),
        ("Vocabulary Constraints", test_vocabulary_selection_constraints()),
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
