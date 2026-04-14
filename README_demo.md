# Clinical Trial Duration Modeling — Interview Guide

## 1. Overview

This project predicts clinical trial duration from structured protocol and trial metadata. It solves planning-time estimation by producing phase-aware duration forecasts rather than a single global estimate. The main output is a regression report with per-phase train/validation/test performance.

## 2. Pipeline (high level)

- Cohort assembly
- Feature construction
- Target definition
- Model training
- Evaluation

## 3. Core files (ordered)

- `4_regression/core/step00_cohort_io.py` — Loads and joins the modeling cohort from preprocessed and raw sources.
- `4_regression/core/step01_features.py` — Builds the model-ready feature matrix and applies feature-policy constraints.
- `4_regression/core/step02_targets.py` — Defines duration target variants and shared target/deviation helpers.
- `4_regression/core/step03_train_regression.py` — Main training entrypoint for dedicated and joint phase regression models.
- `4_regression/core/step04_evaluation.py` — Computes and formats regression/deviation metrics used in reports.

## 4. How to run

```bash
python 4_regression/core/step03_train_regression.py
```

## 5. Key output

- `results/regression_report.txt` is the main output report.

## 6. Suggested walkthrough order

1. `4_regression/core/step00_cohort_io.py`
2. `4_regression/core/step01_features.py`
3. `4_regression/core/step02_targets.py`
4. `4_regression/core/step03_train_regression.py`
5. `4_regression/core/step04_evaluation.py`
6. `results/regression_report.txt`

## 7. Key modeling decisions

- Per-phase models to respect phase-specific duration dynamics.
- `HistGradientBoostingRegressor` as the core regressor.
- Log target transform via `log1p` and inverse `expm1`.
- No feature scaling; numeric NaNs are handled natively by the model.

## 8. Notes

- Features are designed to capture trial design and operational complexity.
