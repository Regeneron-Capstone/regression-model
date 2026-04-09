# Duration and planning-time models

## Completed trials only

All duration modeling uses **COMPLETED** studies (actual start and completion dates available). Preprocessing filters extreme primary spans when building `duration_days` (see `3_preprocessing/preprocess.py` — e.g. **14 ≤ duration_days ≤ 3650**).

## Regression: model family and target transform

- **Regressor:** `sklearn.ensemble.HistGradientBoostingRegressor` (`max_iter=200`, fixed `random_state` where applicable).
- **Target:** `sklearn.compose.TransformedTargetRegressor` with **`np.log1p`** / **`np.expm1`** so the booster fits on log scale; metrics in reports are in **days**.
- **Features:** no global `StandardScaler`; numeric missing values stay **NaN** for HGBR.

## Targets (`4_regression/targets.py`)

| `target_kind` | Definition (days) |
|---------------|-------------------|
| `primary_completion` (default) | primary completion − start (`duration_days` when present) |
| `post_primary_completion` | study completion − primary completion |
| `total_completion` | study completion − start |

Training and reports select the target via CLI `--target` and attach the corresponding column in `prepare_features` / `assemble_feature_matrix`.

## Feature policies

- **`baseline`** — full feature groups used in historical primary-duration training (includes `start_year`, site-footprint fields such as facility counts, etc.). Default for primary and total-completion experiments unless overridden.
- **`strict_planning`** — drops columns listed in `4_regression/feature_registry.py` (planning-safe protocol/design/eligibility signal; **no** `start_year`, **no** operational site-footprint fields). Used for post-primary duration training and late-risk classification.

Column lists for joins are centralized in **`4_regression/cohort_columns.py`**; loading joined tables is **`4_regression/cohort_io.load_and_join`**.

## Regression architecture (`4_regression/train_regression.py`)

Not one global model over all rows. For each target + feature policy:

- **Dedicated** models fit **PHASE1**, **PHASE2**, **PHASE3** separately (constant phase within cohort; phase is **not** one-hot encoded inside those cohorts).
- **Early joint** trains on **PHASE1 ∪ PHASE1/PHASE2 ∪ PHASE2**; used to evaluate **PHASE1/PHASE2** rows in the joint test fold.
- **Late joint** trains on **PHASE2 ∪ PHASE2/PHASE3 ∪ PHASE3**; used for **PHASE2/PHASE3** mixed labels in the joint test fold.

Reports include mixed-cohort baselines and summary tables; see `results/regression_report.txt` (or target-specific report names).

## Train / val / test

**60% / 20% / 20%** splits, `random_state=42` (unless overridden), consistent with deviation analysis and metadata capture.

## Staged full-duration forecast

**`4_regression/combined_duration_forecast.py`** loads (or refits) two stacks of stage models — **primary** with **baseline** features and **post-primary** with **strict_planning** features — then sums components to a **total predicted completion span** and writes a CSV (see script docstring). Persisted bundles live under `results/stage_models/` by default or under an experiment directory when using **`main.py --planning-experiment`** or **`4_regression/planning_experiment_runner.py`**.

## Late-risk classification

**`4_regression/late_risk_classifier.py`** trains **`HistGradientBoostingClassifier`** (`class_weight='balanced'`) on **strict_planning** features with label derived from **total completion** duration vs per-phase training quantiles (see report header). Outputs classification metrics (precision, recall, F1, ROC-AUC, PR-AUC) on val/test.

## Deviation and comparison

- **Deviation:** `5_deviation/deviation_analysis.py` — percent deviation of actual vs predicted for configurable targets; **`--target combined`** consumes the combined forecast CSV.
- **Baseline vs staged summary:** `4_regression/build_final_comparison_report.py` aggregates regression reports, optional **`baseline_metadata.json`**, optional **frozen** primary regression report, and the late-risk report into **`final_comparison_metrics.csv`** and a Markdown (optional TXT) narrative.

## Metrics

Regression: **RMSE**, **MAE**, **R²** on train / val / test as printed in each report. Classification: see late-risk report. For a single-command refresh of the default primary baseline report, run `python 4_regression/train_regression.py` and open **`results/regression_report.txt`**.
