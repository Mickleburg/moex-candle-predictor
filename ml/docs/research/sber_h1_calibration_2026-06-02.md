# SBER H1 — Probability Calibration — 2026-06-02

## Hypothesis
ExtraTrees predict_proba returns uncalibrated scores. Isotonic/Platt calibration
on held-out data will spread probabilities without hurting F1.

## Method
- Model: ExtraTreesClassifier (n=300, depth=None, leaf=20, sqrt) frozen candidate
- Target: triple_barrier:h3:w12:up1.25:down1.25
- Walk-forward: 4 folds, initial_train=12000, val=2000
- Cal set: last 1800 rows of each fold's train window (never seen by ExtraTrees)
- Isotonic: per-class IsotonicRegression (non-parametric, ~600 cal samples per class)
- Platt: per-class LogisticRegression on raw scores (2 params, less overfitting risk)
- Seeds: [7, 42, 100]

## Results — F1 and Calibration Quality

| Metric | Uncalibrated | Isotonic | Platt |
|--------|-------------|----------|-------|
| Val macro-F1 (mean ± std) | 0.4675 ± 0.0160 | 0.4503 ± 0.0347 | 0.4520 ± 0.0200 |
| Worst fold F1 | 0.4400 | 0.3933 | 0.4200 |
| ECE (↓ better) | 0.0448 | 0.0457 | 0.0464 |

## Results — Confidence Coverage

| Threshold | Uncalibrated | Isotonic | Platt |
|-----------|-------------|----------|-------|
| > 0.40 | 58.1% | 82.9% | 73.5% |
| > 0.45 | 25.9% | 50.7% | 39.2% |
| > 0.50 | 16.2% | 30.9% | 22.3% |

## Per-class F1

| Class | Isotonic | Platt |
|-------|----------|-------|
| SELL | 0.3806 | 0.3607 |
| HOLD | 0.5780 | 0.5808 |
| BUY | 0.3922 | 0.4144 |

## Calibration Curves — Isotonic (aggregated val folds, quantile bins)

Ideal: mean_conf == frac_pos (on diagonal). Underconfident: curve below diagonal.

### SELL

| Bin | Uncal conf | Uncal frac_pos | Cal (iso) conf | Cal frac_pos |
|-----|-----------|----------------|---------------|--------------|
| 1 | 0.191 | 0.138 | 0.109 | 0.139 |
| 2 | 0.242 | 0.219 | 0.208 | 0.245 |
| 3 | 0.290 | 0.343 | 0.281 | 0.320 |
| 4 | 0.325 | 0.344 | 0.317 | 0.350 |
| 5 | 0.356 | 0.351 | 0.364 | 0.370 |
| 6 | 0.379 | 0.383 | 0.400 | 0.400 |
| 7 | 0.398 | 0.447 | 0.432 | 0.410 |
| 8 | 0.427 | 0.455 | 0.514 | 0.448 |

### HOLD

| Bin | Uncal conf | Uncal frac_pos | Cal (iso) conf | Cal frac_pos |
|-----|-----------|----------------|---------------|--------------|
| 1 | 0.198 | 0.145 | 0.096 | 0.160 |
| 2 | 0.224 | 0.165 | 0.169 | 0.185 |
| 3 | 0.247 | 0.205 | 0.203 | 0.189 |
| 4 | 0.273 | 0.263 | 0.252 | 0.271 |
| 5 | 0.309 | 0.277 | 0.293 | 0.276 |
| 6 | 0.384 | 0.364 | 0.393 | 0.340 |
| 7 | 0.486 | 0.585 | 0.565 | 0.594 |
| 8 | 0.586 | 0.741 | 0.756 | 0.739 |

### BUY

| Bin | Uncal conf | Uncal frac_pos | Cal (iso) conf | Cal frac_pos |
|-----|-----------|----------------|---------------|--------------|
| 1 | 0.216 | 0.112 | 0.121 | 0.120 |
| 2 | 0.266 | 0.197 | 0.208 | 0.181 |
| 3 | 0.305 | 0.302 | 0.290 | 0.298 |
| 4 | 0.345 | 0.332 | 0.348 | 0.346 |
| 5 | 0.364 | 0.395 | 0.376 | 0.407 |
| 6 | 0.378 | 0.401 | 0.402 | 0.367 |
| 7 | 0.393 | 0.384 | 0.427 | 0.411 |
| 8 | 0.420 | 0.452 | 0.489 | 0.447 |

## Conclusion and Decision

**Isotonic**: F1 -0.0172, ECE +0.0010, conf>0.5: 16.2% → 30.9%
**Platt**:    F1 -0.0155, ECE +0.0016, conf>0.5: 16.2% → 22.3%

Neither method improves ECE significantly while preserving F1.

**Key finding**: Uncalibrated ECE=0.0448 is already reasonable for a 3-class problem.
ExtraTrees probabilities are not as miscalibrated as expected.

**Decision**: Keep uncalibrated model. Add calibration only as optional post-processing step
in risk_manager when threshold-based filtering is needed.
Set `probabilities_calibrated: false` in artifact metadata (truthful).

Better path: improve signal quality first (Word2Vec embeddings, Step 3),
then calibrate the stronger model.

Next step: Step 3 — Word2Vec candle embeddings (primary quality improvement).