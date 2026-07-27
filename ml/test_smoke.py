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


def test_target_feature_research_invariants():
    """Test alternative targets and continuous features stay past-only/aligned."""

    print("\nTesting target/continuous feature research invariants...")

    try:
        import numpy as np

        from src.data.fixtures import generate_mock_candles
        from src.nlp.action_features import make_continuous_past_features, standardize_by_train
        from src.nlp.targets import ActionTargetSpec, make_research_action_targets, past_return_volatility, triple_barrier_details
        from sber_action_target_feature_research import (
            ModelConfig,
            aggregate_rows,
            build_feature_set_matrices,
            build_model_configs,
            continuous_feature_mask,
            economic_sanity,
            fit_predict_model,
            target_audit_summary,
        )

        df = generate_mock_candles(n=160, ticker="SBER", timeframe="1H", seed=42)
        ret_lo = make_research_action_targets(df, ActionTargetSpec(mode="return_threshold", horizon=1, return_threshold_mult=0.75))
        ret_hi = make_research_action_targets(df, ActionTargetSpec(mode="return_threshold", horizon=1, return_threshold_mult=1.5))
        assert ret_lo.threshold < ret_hi.threshold
        assert np.count_nonzero(ret_hi.labels == 1) >= np.count_nonzero(ret_lo.labels == 1)

        vol_target = make_research_action_targets(
            df,
            ActionTargetSpec(mode="volatility_adjusted_return", horizon=3, vol_window=16, vol_k=1.0),
        )
        assert vol_target.effective_horizon == 3
        assert np.all(vol_target.labels[-3:] == -1)
        assert "past_volatility" not in make_continuous_past_features.__code__.co_varnames

        barrier = make_research_action_targets(
            df,
            ActionTargetSpec(mode="triple_barrier", barrier_horizon=6, barrier_vol_window=16, barrier_up_k=1.0, barrier_down_k=1.0),
        )
        assert barrier.effective_horizon == 6
        assert np.all(barrier.labels[-6:] == -1)
        barrier_spec = ActionTargetSpec(mode="triple_barrier", barrier_horizon=6, barrier_vol_window=16, barrier_up_k=1.0, barrier_down_k=1.0)
        details = triple_barrier_details(df, barrier_spec)
        idx = np.arange(32, 60)
        assert np.allclose(details["upper_barrier"][idx], details["close"][idx] * (1.0 + details["upper_return"][idx]))
        assert np.allclose(details["lower_barrier"][idx], details["close"][idx] * (1.0 - details["lower_return"][idx]))
        assert np.allclose(details["past_volatility"][:80], triple_barrier_details(df.iloc[:100].copy(), barrier_spec)["past_volatility"][:80])
        audit = target_audit_summary(details, idx, barrier.labels[idx])
        assert "share_upper_first" in audit and "mean_mfe" in audit["by_label"]["BUY"]
        econ = economic_sanity(barrier.labels[idx], barrier.labels[idx], idx, barrier.future_returns, details)
        assert "mean_realized_return_by_prediction" in econ
        assert "predicted_action_barrier_hit_rate" in econ

        features, names = make_continuous_past_features(df)
        assert features.shape[0] == len(df)
        assert features.shape[1] == len(names)
        assert np.all(np.isfinite(features))
        full_mask = continuous_feature_mask("lm_regime_continuous", names)
        no_session = continuous_feature_mask("lm_regime_continuous_no_session", names)
        no_volume = continuous_feature_mask("lm_regime_continuous_no_volume", names)
        assert full_mask.sum() == len(names)
        assert no_session.sum() < full_mask.sum()
        assert no_volume.sum() < full_mask.sum()

        train_idx = np.arange(32, 100)
        val_idx = np.arange(100, 130)
        X_val = standardize_by_train(features, train_idx, val_idx)
        assert X_val.shape == (len(val_idx), features.shape[1])
        assert np.all(np.isfinite(X_val))

        train_small = np.arange(32, 80)
        val_small = np.arange(80, 110)
        X_train_small = standardize_by_train(features, train_small, train_small)
        X_val_small = standardize_by_train(features, train_small, val_small)
        y_train_small = ret_lo.labels[train_small]
        valid = y_train_small >= 0
        pred, proba, diag = fit_predict_model(
            X_train_small[valid],
            y_train_small[valid],
            X_val_small,
            model_config=ModelConfig("hist_gb", "hist_gb:test", {"max_iter": 5, "max_leaf_nodes": 7}),
            class_weight="balanced",
            random_state=42,
        )
        assert pred.shape == (len(val_small),)
        assert proba is not None and np.allclose(proba.sum(axis=1), 1.0)
        assert isinstance(diag, dict)

        df_changed = df.copy()
        df_changed.loc[120:, "close"] = df_changed.loc[120:, "close"] * 3.0
        vol_before = past_return_volatility(df, 16)
        vol_after = past_return_volatility(df_changed, 16)
        assert np.allclose(vol_before[:100], vol_after[:100])

        features_after, _ = make_continuous_past_features(df_changed)
        assert np.allclose(features[:100], features_after[:100])
        class Args:
            models = "logreg"
            logreg_c_values = "0.3,1.0"
            logreg_penalties = "l1,l2"
            logreg_solvers = "lbfgs,liblinear"
            hist_gb_learning_rates = "0.05"
            hist_gb_max_leaf_nodes = "31"
            hist_gb_l2 = "0.0"
            hist_gb_max_iter = "10"
            extra_trees_max_depths = "none"
            extra_trees_min_samples_leaf = "5"
            extra_trees_max_features = "sqrt"
            extra_trees_n_estimators = 10

        configs = build_model_configs(Args())
        labels = {config.label for config in configs}
        assert "logreg:C=0.3:penalty=l1:solver=liblinear" in labels
        assert "logreg:C=0.3:penalty=l1:solver=lbfgs" not in labels

        lm_train = np.zeros((4, 18))
        regime_train = np.zeros((4, 17))
        cont_train = np.zeros((4, len(names)))
        X_a, _, _ = build_feature_set_matrices(
            "lm_regime_continuous",
            lm_train=lm_train,
            lm_calib=lm_train,
            lm_val=lm_train,
            regime_train=regime_train,
            regime_calib=regime_train,
            regime_val=regime_train,
            cont_train=cont_train,
            cont_calib=cont_train,
            cont_val=cont_train,
            continuous_names=names,
        )
        X_b, _, _ = build_feature_set_matrices(
            "lm_regime_continuous_no_session",
            lm_train=lm_train,
            lm_calib=lm_train,
            lm_val=lm_train,
            regime_train=regime_train,
            regime_calib=regime_train,
            regime_val=regime_train,
            cont_train=cont_train,
            cont_calib=cont_train,
            cont_val=cont_train,
            continuous_names=names,
        )
        assert X_b.shape[1] < X_a.shape[1]
        aggregate = aggregate_rows(
            [
                {
                    "target_label": "a",
                    "target_mode": "triple_barrier",
                    "feature_set": "continuous_regime",
                    "model": "extra_trees",
                    "class_weight": "balanced",
                    "fold_id": 1,
                    "random_state": 42,
                    "prediction_distribution": {"BUY": {"share": 0.2}, "SELL": {"share": 0.3}, "HOLD": {"share": 0.5}},
                    "metrics": {
                        "macro_f1": 0.4,
                        "accuracy": 0.5,
                        "balanced_accuracy": 0.4,
                        "buy_f1": 0.3,
                        "sell_f1": 0.4,
                        "hold_f1": 0.5,
                        "action_rate": 0.5,
                        "hold_rate": 0.5,
                        "buy_sell_hmean_f1": 0.34,
                    },
                },
                {
                    "target_label": "a",
                    "target_mode": "triple_barrier",
                    "feature_set": "continuous_regime",
                    "model": "extra_trees",
                    "class_weight": "balanced",
                    "fold_id": 2,
                    "random_state": 42,
                    "prediction_distribution": {"BUY": {"share": 0.25}, "SELL": {"share": 0.25}, "HOLD": {"share": 0.5}},
                    "metrics": {
                        "macro_f1": 0.6,
                        "accuracy": 0.6,
                        "balanced_accuracy": 0.6,
                        "buy_f1": 0.5,
                        "sell_f1": 0.6,
                        "hold_f1": 0.7,
                        "action_rate": 0.5,
                        "hold_rate": 0.5,
                        "buy_sell_hmean_f1": 0.54,
                    },
                },
            ]
        )[0]
        assert aggregate["mean_macro_f1"] == 0.5
        assert aggregate["worst_macro_f1"] == 0.4
        print("  PASS Alternative targets and continuous features")
        return True
    except Exception as exc:
        print(f"  FAIL Target/feature research invariant test failed: {exc}")
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
            aggregate_rows,
            build_regime_feature_matrix,
            nested_range,
            objective_value,
            parse_int_list,
            resolve_class_weight,
            select_global_thresholds,
            select_regime_thresholds,
            threshold_predictions,
        )
        from sber_action_final_evaluation import final_nested_range, parse_random_state_policy
        from src.data.fixtures import generate_mock_candles
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
        decision = select_global_thresholds(
            y,
            proba,
            grid,
            temps,
            selection_objective="macro_f1",
            temperature_selection="objective",
            target_action_rate=0.5,
            action_rate_penalty=0.1,
            mode="global",
        )
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
            selection_objective="macro_f1",
            temperature_selection="objective",
            target_action_rate=0.5,
            action_rate_penalty=0.1,
            mode="regime_test",
            min_regime_calibration_samples=3,
        )
        assert regime_decision.regime_thresholds is not None
        assert regime_decision.regime_thresholds["a"]["fallback"]
        assert regime_decision.regime_thresholds["b"]["fallback"]

        oracle = select_global_thresholds(
            y,
            proba,
            grid,
            temps,
            selection_objective="macro_f1",
            temperature_selection="objective",
            target_action_rate=0.5,
            action_rate_penalty=0.1,
            mode="oracle_global",
            is_oracle=True,
        )
        assert oracle.is_oracle
        metrics = {"macro_f1": 0.4, "buy_f1": 0.3, "sell_f1": 0.2, "buy_sell_mean_f1": 0.25, "action_rate": 0.7}
        assert objective_value(metrics, "buy_sell_mean_f1", target_action_rate=0.5, action_rate_penalty=0.1) == 0.25
        assert objective_value(metrics, "macro_f1_action_penalty", target_action_rate=0.5, action_rate_penalty=0.1) < metrics["macro_f1"]
        assert resolve_class_weight("action_boost_1.2") == {0: 1.2, 1: 0.8, 2: 1.2}
        assert parse_int_list("7,13,42") == [7, 13, 42]
        assert parse_random_state_policy("fixed:42") == 42

        final_range, holdout = final_nested_range(1000, calibration_size=100)
        assert holdout["test"][0] == 850
        assert final_range.inner_train_end == 750
        assert final_range.calibration_start == 750
        assert final_range.calibration_end == 850
        assert final_range.outer_fold.val_start == 850

        aggregate = aggregate_rows(
            [
                {
                    "vocabulary": "shape/gmm_diag/20",
                    "feature_set": "lm_regime",
                    "classifier": "logreg",
                    "class_weight": "action_boost_1.2",
                    "action_horizon": 1,
                    "threshold_mode": "argmax",
                    "selection_objective": "argmax",
                    "temperature_selection": "none",
                    "calibration_method": "none",
                    "is_oracle": False,
                    "outer_fold_id": 1,
                    "random_state": 7,
                    "metrics": {
                        "macro_f1": 0.4,
                        "buy_f1": 0.3,
                        "sell_f1": 0.2,
                        "hold_f1": 0.5,
                        "buy_sell_mean_f1": 0.25,
                        "buy_sell_hmean_f1": 0.24,
                        "min_buy_sell_f1": 0.2,
                        "action_rate": 0.6,
                    },
                    "prediction_distribution": {"BUY": {"share": 0.3}, "SELL": {"share": 0.3}, "HOLD": {"share": 0.4}},
                },
                {
                    "vocabulary": "shape/gmm_diag/20",
                    "feature_set": "lm_regime",
                    "classifier": "logreg",
                    "class_weight": "action_boost_1.2",
                    "action_horizon": 1,
                    "threshold_mode": "argmax",
                    "selection_objective": "argmax",
                    "temperature_selection": "none",
                    "calibration_method": "none",
                    "is_oracle": False,
                    "outer_fold_id": 1,
                    "random_state": 13,
                    "metrics": {
                        "macro_f1": 0.42,
                        "buy_f1": 0.31,
                        "sell_f1": 0.21,
                        "hold_f1": 0.52,
                        "buy_sell_mean_f1": 0.26,
                        "buy_sell_hmean_f1": 0.25,
                        "min_buy_sell_f1": 0.21,
                        "action_rate": 0.62,
                    },
                    "prediction_distribution": {"BUY": {"share": 0.32}, "SELL": {"share": 0.3}, "HOLD": {"share": 0.38}},
                },
            ]
        )[0]
        assert aggregate["random_states"] == [7, 13]
        assert aggregate["macro_f1_std_across_seeds"] > 0

        df = generate_mock_candles(n=80, ticker="SBER", timeframe="1H", seed=42)
        train_indices = np.arange(20, 50)
        target_indices = np.arange(50, 65)
        lm_train = np.zeros((len(train_indices), 18), dtype=float)
        lm_target = np.zeros((len(target_indices), 18), dtype=float)
        lm_train[:, 0] = np.linspace(0.2, 0.8, len(train_indices))
        lm_train[:, 2] = np.linspace(0.4, 0.9, len(train_indices))
        lm_train[:, 3] = np.linspace(0.1, 1.0, len(train_indices))
        lm_target[:, 0] = 0.5
        lm_target[:, 2] = 0.7
        lm_target[:, 3] = np.linspace(0.2, 0.8, len(target_indices))
        regime_features = build_regime_feature_matrix(df, train_indices, target_indices, lm_train, lm_target)
        assert regime_features.shape[0] == len(target_indices)
        assert np.all(np.isfinite(regime_features))
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


def test_ml_prediction_contract_invariants():
    """Test candle_batch -> ml_prediction contract helpers."""

    print("\nTesting ML prediction JSON contract invariants...")

    try:
        import json
        import math
        import pickle
        import tempfile
        from datetime import datetime, timedelta

        import numpy as np
        from sklearn.dummy import DummyClassifier

        from src.nlp import make_continuous_past_features
        from src.service.contracts import (
            build_artifact_missing_response,
            build_ml_prediction_response,
            candle_batch_to_dataframe,
            load_candle_batch_json,
        )
        from src.service.research_artifact import build_artifact_prediction_response, load_research_artifact, predict_with_artifact

        base = datetime(2026, 5, 15, 10)
        payload = {
            "ticker": "SBER",
            "timeframe": "1H",
            "candles": [
                {"begin": (base + timedelta(hours=2)).isoformat(), "open": 300, "high": 302, "low": 299, "close": 301, "volume": 100},
                {"begin": base.isoformat(), "open": 298, "high": 301, "low": 297, "close": 300, "volume": 120},
                {"begin": (base + timedelta(hours=1)).isoformat(), "open": 300, "high": 303, "low": 299, "close": 302, "volume": 110},
            ],
        }
        batch = load_candle_batch_json(payload)
        df = candle_batch_to_dataframe(batch)
        assert df["begin"].is_monotonic_increasing
        assert df["ticker"].nunique() == 1
        assert df["timeframe"].nunique() == 1

        duplicate = dict(payload)
        duplicate["candles"] = [payload["candles"][0], payload["candles"][0]]
        try:
            candle_batch_to_dataframe(load_candle_batch_json(duplicate))
            raise AssertionError("duplicate begin was accepted")
        except ValueError:
            pass

        response = build_ml_prediction_response(
            ticker="SBER",
            timeframe="1H",
            as_of=df["begin"].iloc[-1].isoformat(),
            probabilities={"buy": 0.2, "hold": 0.5, "sell": 0.3},
            confidence=0.5,
        )
        assert set(response["probabilities"]) == {"buy", "hold", "sell"}
        assert math.isclose(sum(response["probabilities"].values()), 1.0, abs_tol=1e-6)
        assert response["diagnostics"]["feature_set"] == "continuous_regime"

        missing = build_artifact_missing_response(batch=batch, df=df, artifact_dir="missing")
        assert missing["diagnostics"]["artifact_missing"] is True
        assert missing["probabilities"] == {"buy": 0.0, "hold": 1.0, "sell": 0.0}
        assert missing["confidence"] == 0.0
        json.dumps(missing)

        feature_matrix, feature_names = make_continuous_past_features(df)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            model = DummyClassifier(strategy="prior")
            model.fit(np.zeros((3, len(feature_names))), np.asarray([0, 1, 2], dtype=int))
            with (tmp_path / "model.pkl").open("wb") as handle:
                pickle.dump(model, handle)
            (tmp_path / "metadata.json").write_text(
                json.dumps(
                    {
                        "artifact_id": "smoke",
                        "model_version": "smoke",
                        "artifact_type": "research",
                        "is_production": False,
                        "model_family": "triple_barrier_extra_trees",
                        "target": "triple_barrier:h3:w12:up1.25:down1.25",
                        "feature_set": "continuous_regime",
                        "class_weight": "none",
                        "validation_macro_f1_mean": 0.1,
                        "min_candles_for_prediction": 2,
                    }
                ),
                encoding="utf-8",
            )
            (tmp_path / "feature_config.json").write_text(
                json.dumps(
                    {
                        "feature_set": "continuous_regime",
                        "feature_columns": feature_names,
                        "standardization_mean": [0.0] * len(feature_names),
                        "standardization_std": [1.0] * len(feature_names),
                    }
                ),
                encoding="utf-8",
            )
            (tmp_path / "target_config.json").write_text(
                json.dumps({"target_mode": "triple_barrier", "label_order": ["SELL", "HOLD", "BUY"]}),
                encoding="utf-8",
            )
            (tmp_path / "label_mapping.json").write_text(
                json.dumps(
                    {
                        "internal_to_contract": {"SELL": "sell", "HOLD": "hold", "BUY": "buy"},
                        "contract_to_internal": {"sell": "SELL", "hold": "HOLD", "buy": "BUY"},
                    }
                ),
                encoding="utf-8",
            )
            (tmp_path / "schema_version.json").write_text(json.dumps({"artifact_schema_version": 1}), encoding="utf-8")
            (tmp_path / "feature_columns.json").write_text(json.dumps(feature_names), encoding="utf-8")

            artifact = load_research_artifact(tmp_path)
            artifact_response = build_artifact_prediction_response(batch=batch, df=df, artifact=artifact)
            assert artifact_response["diagnostics"]["artifact_missing"] is False
            assert artifact_response["diagnostics"]["is_production"] is False
            assert math.isclose(sum(artifact_response["probabilities"].values()), 1.0, abs_tol=1e-6)

            # New contract info: expected_return + signal_context (informational, not commands)
            assert isinstance(artifact_response["expected_return"], float)
            ctx = artifact_response["signal_context"]
            assert ctx["horizon_bars"] == 3
            assert ctx["horizon_timeframe"] == "1H"
            assert ctx["upper_return"] >= 0.001 and ctx["lower_return"] >= 0.001
            assert ctx["upper_barrier"] > ctx["reference_close"] > ctx["lower_barrier"]
            assert ctx["calibrated"] is False
            json.dumps(artifact_response)

            insufficient = predict_with_artifact(artifact, df.head(1))
            assert insufficient.diagnostics["error"] == "insufficient_history"
            assert insufficient.probabilities == {"buy": 0.0, "hold": 1.0, "sell": 0.0}
        print("  PASS ML prediction contract helpers")
        return True
    except Exception as exc:
        print(f"  FAIL ML prediction contract test failed: {exc}")
        return False


def test_timezone_canonicalization():
    """Test MSK timezone fix preserves wall-clock hour/dow while correcting the tz label."""

    print("\nTesting timezone canonicalization...")

    try:
        import pandas as pd

        from src.data.load import MOEX_TZ, to_moscow_time

        # Legacy: MSK wall-clock mislabelled as UTC.
        legacy = pd.Series(pd.to_datetime(
            ["2020-01-03 09:00:00", "2025-03-14 22:00:00"], utc=True))
        fixed = to_moscow_time(legacy)
        assert str(fixed.dt.tz) == MOEX_TZ
        # Wall-clock hour/dow unchanged -> models stay valid.
        assert list(fixed.dt.hour) == list(legacy.dt.hour) == [9, 22]
        assert list(fixed.dt.dayofweek) == list(legacy.dt.dayofweek)
        # Correct label is +03:00.
        assert fixed.iloc[0].utcoffset().total_seconds() == 3 * 3600

        # Naive input is localised, not shifted.
        naive = pd.Series(pd.to_datetime(["2025-03-14 22:00:00"]))
        fixed_naive = to_moscow_time(naive)
        assert list(fixed_naive.dt.hour) == [22]
        assert str(fixed_naive.dt.tz) == MOEX_TZ
        print("  PASS Timezone canonicalization (wall-clock preserved, label corrected to MSK)")
        return True
    except Exception as exc:
        print(f"  FAIL Timezone canonicalization test failed: {exc}")
        return False


def test_orthogonal_tz_alignment():
    """Orthogonal merge_asof must handle mismatched tz (contract +03:00 vs named MSK)."""

    print("\nTesting orthogonal tz alignment...")

    try:
        import pandas as pd

        from src.features.orthogonal import _align_backward

        # Target candle begins as a fixed +03:00 offset (as parsed from a contract isoformat)
        target = pd.to_datetime(["2025-03-14 22:00:00+03:00", "2025-03-14 23:00:00+03:00"])
        # Orthogonal feature bars carry a NAMED Europe/Moscow tz (different tz object)
        feats = pd.DataFrame({
            "begin": pd.to_datetime(["2025-03-14 21:00:00", "2025-03-14 22:00:00"]).tz_localize("Europe/Moscow"),
            "X_ret_1h": [0.01, 0.02],
        })
        merged = _align_backward(pd.Series(target), feats)  # must not raise on mismatched tz
        # backward: 22:00 -> 22:00 bar (0.02); 23:00 -> last <=23:00 is 22:00 bar (0.02)
        assert list(merged["X_ret_1h"]) == [0.02, 0.02], list(merged["X_ret_1h"])
        print("  PASS Orthogonal tz alignment (mismatched tz merged, backward semantics)")
        return True
    except Exception as exc:
        print(f"  FAIL Orthogonal tz alignment test failed: {exc}")
        return False


def test_ticker_model_router():
    """Test per-ticker routing: known ticker -> model, unknown ticker -> artifact_missing."""

    print("\nTesting per-ticker model router...")

    try:
        import json
        import pickle
        import tempfile
        from datetime import datetime, timedelta

        import numpy as np
        from sklearn.dummy import DummyClassifier

        from src.nlp import make_continuous_past_features
        from src.service.contracts import candle_batch_to_dataframe, load_candle_batch_json
        from src.service.model_registry import TickerModelRouter, resolve_artifact_dir

        base = datetime(2026, 5, 15, 10)
        candles = [
            {"begin": (base + timedelta(hours=i)).isoformat(), "open": 300 + i, "high": 303 + i,
             "low": 298 + i, "close": 301 + i, "volume": 100 + i}
            for i in range(40)
        ]

        def make_batch(ticker):
            return load_candle_batch_json({"ticker": ticker, "timeframe": "1H", "candles": candles})

        df_for_names = candle_batch_to_dataframe(make_batch("SBER"))
        _, feature_names = make_continuous_past_features(df_for_names)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Build a minimal ET artifact for SBER under the ET template name.
            art = root / "research_triple_barrier_sber_h1"
            art.mkdir(parents=True)
            model = DummyClassifier(strategy="prior")
            model.fit(np.zeros((3, len(feature_names))), np.asarray([0, 1, 2], dtype=int))
            with (art / "model.pkl").open("wb") as handle:
                pickle.dump(model, handle)
            (art / "metadata.json").write_text(json.dumps({
                "artifact_id": "smoke_sber", "model_version": "smoke_sber", "ticker": "SBER",
                "timeframe": "1H", "model_family": "triple_barrier_extra_trees",
                "target": "triple_barrier:h3:w12:up1.25:down1.25", "feature_set": "continuous_regime",
                "class_weight": "none", "validation_macro_f1_mean": 0.1, "min_candles_for_prediction": 2,
            }), encoding="utf-8")
            (art / "feature_config.json").write_text(json.dumps({
                "feature_set": "continuous_regime", "feature_columns": feature_names,
                "standardization_mean": [0.0] * len(feature_names),
                "standardization_std": [1.0] * len(feature_names),
            }), encoding="utf-8")
            (art / "target_config.json").write_text(json.dumps(
                {"target_mode": "triple_barrier", "label_order": ["SELL", "HOLD", "BUY"]}), encoding="utf-8")
            (art / "label_mapping.json").write_text(json.dumps({
                "internal_to_contract": {"SELL": "sell", "HOLD": "hold", "BUY": "buy"},
                "contract_to_internal": {"sell": "SELL", "hold": "HOLD", "buy": "BUY"},
            }), encoding="utf-8")
            (art / "schema_version.json").write_text(json.dumps({"artifact_schema_version": 1}), encoding="utf-8")
            (art / "feature_columns.json").write_text(json.dumps(feature_names), encoding="utf-8")

            router = TickerModelRouter(artifact_root=root)
            assert router.available_tickers() == ["SBER"], router.available_tickers()
            assert resolve_artifact_dir("SBER", root) == art
            assert resolve_artifact_dir("GAZP", root) is None

            sber_resp = router.predict(make_batch("SBER"))
            assert sber_resp["diagnostics"]["artifact_missing"] is False
            assert sber_resp["ticker"] == "SBER"

            gazp_resp = router.predict(make_batch("GAZP"))
            assert gazp_resp["diagnostics"]["artifact_missing"] is True
            assert gazp_resp["ticker"] == "GAZP"
            json.dumps(gazp_resp)
        print("  PASS Per-ticker routing (SBER->model, GAZP->artifact_missing)")
        return True
    except Exception as exc:
        print(f"  FAIL Ticker model router test failed: {exc}")
        return False


def test_dividend_sleeve_max_weight_small_book():
    """1a: inverse-vol cap is a HARD per-name limit even on 1-2 name books. A single cap+renorm pass
    breached it (1 name -> 1.0; 2 equal -> 0.5/0.5); fixpoint water-filling holds MAX_WEIGHT and leaves
    a too-small book intentionally under-invested (gross < 1). Feasible books (n*cap>=1) sum to 1."""

    print("\nTesting dividend-sleeve MAX_WEIGHT on small books...")
    try:
        import numpy as np
        import pandas as pd

        from src.service.dividend_sleeve import MAX_WEIGHT, inverse_vol_weights

        rng = np.random.default_rng(0)
        idx = pd.date_range("2025-01-01", periods=40, freq="D", tz="Europe/Moscow")
        cols = ["A", "B", "C", "D"]
        prices = pd.DataFrame(100 + np.cumsum(rng.normal(0.0, 1.0, size=(40, 4)), axis=0),
                              index=idx, columns=cols)
        pos = len(prices) - 1

        for names in (["A"], ["A", "B"]):
            w = inverse_vol_weights(prices, names, pos)
            assert w, f"empty weights for {names}"
            assert all(x <= MAX_WEIGHT + 1e-9 for x in w.values()), f"MAX_WEIGHT breached on {names}: {w}"
            assert 0.0 < sum(w.values()) <= len(names) * MAX_WEIGHT + 1e-9, \
                f"gross should not exceed n*cap on small book {names}: {w}"
        # 1-name book pins exactly at the cap (old single-pass renorm returned 1.0)
        assert abs(inverse_vol_weights(prices, ["A"], pos)["A"] - MAX_WEIGHT) < 1e-9

        # feasible book (n*cap >= 1): cap still respected AND fully invested (sum ~ 1)
        w4 = inverse_vol_weights(prices, cols, pos)
        assert all(x <= MAX_WEIGHT + 1e-9 for x in w4.values()), f"MAX_WEIGHT breached: {w4}"
        assert abs(sum(w4.values()) - 1.0) < 1e-6, f"feasible book should sum to 1: {sum(w4.values())}"
        print("  PASS MAX_WEIGHT held on 1/2-name books; feasible book sums to 1")
        return True
    except Exception as exc:
        print(f"  FAIL Dividend-sleeve MAX_WEIGHT test failed: {exc}")
        raise


def test_h9_daily_series_excludes_weekend_sessions():
    """H9 counts entry/exit offsets as index POSITIONS, so a bar MUST equal a trading day. MOEX runs
    weekend sessions since 2025, which silently turned the deployed '-12 trading days' into -12 bars
    = ~8 real TD on 2026 events while the in-sample benchmark (<2025) stayed a true 12 TD — the gate
    would then compare the forward against a different rule. Both loaders must drop Sat/Sun.
    See ml/docs/research/h9_weekend_bar_drift_prereg_2026-07-28.md."""

    print("\nTesting H9 daily series excludes MOEX weekend sessions...")
    try:
        import pandas as pd

        from scripts.h9_dividend_research import load_daily
        from src.features.cross_sectional import _drop_weekend_sessions

        # unit: the shared helper drops Sat/Sun and keeps every weekday
        idx = pd.date_range("2026-07-01", "2026-07-20", freq="D", tz="Europe/Moscow")
        s = pd.Series(range(len(idx)), index=idx)
        out = _drop_weekend_sessions(s)
        assert not out.empty, "helper dropped everything"
        assert (out.index.dayofweek < 5).all(), "weekend bar survived the helper"
        assert len(out) == int((idx.dayofweek < 5).sum()), "helper dropped a weekday"

        # integration on the real panel (skipped when data/raw isn't present, e.g. clean CI)
        checked = 0
        for ticker in ("SBER", "PLZL", "IMOEX"):
            real = load_daily(ticker)
            if real is None or real.empty:
                continue
            checked += 1
            weekend = real.index[real.index.dayofweek >= 5]
            assert len(weekend) == 0, \
                f"{ticker}: {len(weekend)} weekend bars survived (first {weekend[:3].tolist()})"
        print(f"  PASS weekend sessions excluded (helper + {checked} real series)")
        return True
    except Exception as exc:
        print(f"  FAIL H9 weekend-session exclusion test failed: {exc}")
        raise


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
        ("Target/Feature Research", test_target_feature_research_invariants()),
        ("Action LM Robustness", test_action_lm_robustness_invariants()),
        ("Nested Thresholds", test_nested_threshold_invariants()),
        ("Vocabulary Constraints", test_vocabulary_selection_constraints()),
        ("Predictor Validation", test_predictor_input_validation()),
        ("ML Prediction Contract", test_ml_prediction_contract_invariants()),
        ("Timezone Canonicalization", test_timezone_canonicalization()),
        ("Orthogonal TZ Alignment", test_orthogonal_tz_alignment()),
        ("Ticker Model Router", test_ticker_model_router()),
        ("Dividend Sleeve MAX_WEIGHT", test_dividend_sleeve_max_weight_small_book()),
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
