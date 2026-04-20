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

- `6_results/regression_report.txt` is the main output report.

## 5b. Latest verified results

Most recent run in this workspace (condition mapping + `preprocess.py` + `step03_train_regression.py` + `late_risk_classifier.py`). Full copy: **`final_results_capstone.txt`**.

- Preprocessed filtered cohort: **84,879** trials
- Completed modeling cohort: **57,865** trials
- Condition mapping coverage in completed cohort (`has_ccsr=1`): **73.9%**

Primary-completion regression — test metrics (days):

| Phase | R² | RMSE | MAE |
|---|---:|---:|---:|
| PHASE1 (dedicated) | 0.5981 | 331 | 183 |
| PHASE2 (dedicated) | 0.4206 | 481 | 323 |
| PHASE3 (dedicated) | 0.4189 | 453 | 301 |
| PHASE1/PHASE2 (early joint route) | 0.3571 | 606 | 419 |
| PHASE2/PHASE3 (late joint route) | 0.2465 | 576 | 380 |

Late-risk classifier — test split (decision threshold **0.6**): precision **0.5444**, recall **0.4759**, F1 **0.5079**, ROC-AUC **0.7771**, PR-AUC **0.5741**.

Top permutation-importance features (normalized; full list in `6_results/feature_importance_rankings.txt`):

| Rank | Baseline regression (R²) | Late-risk classifier (ROC-AUC) |
|:---:|---|---|
| 1 | `max_planned_followup_days` (0.160) | `max_planned_followup_days` (0.208) |
| 2 | `start_year` (0.058) | `category_NEO` (0.116) |
| 3 | `category_NEO` (0.045) | `enrollment` (0.082) |

## 6. Regression R²: Paper vs Ours (test set)

| Phase | TrialBench (Chen et al., 2025) | This project |
|---|---:|---:|
| Phase 1 | 0.6514 ± 0.0085 | 0.5981 |
| Phase 2 | 0.4125 ± 0.0081 | 0.4206 |
| Phase 3 | 0.3148 ± 0.0085 | 0.4189 |

Interpretation: on this benchmark setup, our simpler tabular pipeline (phase-aware routing + engineered trial design/operational features + HGBR) is comparable on Phase 2 and stronger on Phase 3 versus the reported TrialBench baseline, despite TrialBench using richer multi-modal inputs.

## 7. Walkthrough order

1. `4_regression/core/step00_cohort_io.py`
2. `4_regression/core/step01_features.py`
3. `4_regression/core/step02_targets.py`
4. `4_regression/core/step03_train_regression.py`
5. `4_regression/core/step04_evaluation.py`
6. `6_results/regression_report.txt`

## 8. Key modeling decisions

- Per-phase models to respect phase-specific duration dynamics.
- `HistGradientBoostingRegressor` as the core regressor.
- Log target transform via `log1p` and inverse `expm1`.
- No feature scaling; numeric NaNs are handled natively by the model.

## 9. Notes

- Features are designed to capture trial design and operational complexity.
- Late-risk classifier uses a probability **cutoff of 0.6** for positive predictions (adjustable via `--decision-threshold`).
- Rank feature importance (permutation) via `PYTHONPATH=4_regression python 4_regression/experiments/feature_importance_ranking.py` → `6_results/feature_importance_rankings.txt`.
