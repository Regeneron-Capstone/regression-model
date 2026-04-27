# Duration and planning-time models

## Completed trials only

All duration modeling uses **COMPLETED** studies (actual start and completion dates available). Preprocessing filters extreme primary spans when building `duration_days` (see `3_preprocessing/preprocess.py` — e.g. **14 ≤ duration_days ≤ 3650**).

## Regression: model family and target transform

- **Regressor:** `sklearn.ensemble.HistGradientBoostingRegressor` (`max_iter=200`, fixed `random_state` where applicable).
- **Target:** `sklearn.compose.TransformedTargetRegressor` with **`np.log1p`** / **`np.expm1`** so the booster fits on log scale; metrics in reports are in **days**.
- **Features:** no global `StandardScaler`; numeric missing values stay **NaN** for HGBR.

## Targets (`4_regression/core/step02_targets.py`)

| `target_kind` | Definition (days) |
|---------------|-------------------|
| `primary_completion` (default) | primary completion − start (`duration_days` when present) |
| `post_primary_completion` | study completion − primary completion |
| `total_completion` | study completion − start |

Training and reports select the target via CLI `--target` and attach the corresponding column in `prepare_features` / `assemble_feature_matrix`.

## Feature policies

- **`baseline`** — full feature groups used in historical primary-duration training (includes `start_year`, site-footprint fields such as facility counts, etc.). Default for primary and total-completion experiments unless overridden.
- **`strict_planning`** — drops columns listed in `4_regression/feature_registry.py` (planning-safe protocol/design/eligibility signal; **no** `start_year`, **no** operational site-footprint fields). Used for post-primary duration training and late-risk classification.

Column lists for joins are centralized in **`4_regression/cohort_columns.py`**; loading joined tables is **`4_regression/core/step00_cohort_io.load_and_join`**.

## Regression architecture (`4_regression/core/step03_train_regression.py`)

Not one global model over all rows. For each target + feature policy:

- **Dedicated** models fit **PHASE1**, **PHASE2**, **PHASE3** separately (constant phase within cohort; phase is **not** one-hot encoded inside those cohorts).
- **Early joint** trains on **PHASE1 ∪ PHASE1/PHASE2 ∪ PHASE2**; used to evaluate **PHASE1/PHASE2** rows in the joint test fold.
- **Late joint** trains on **PHASE2 ∪ PHASE2/PHASE3 ∪ PHASE3**; used for **PHASE2/PHASE3** mixed labels in the joint test fold.

Reports include mixed-cohort baselines and summary tables; see `6_results/regression_report.txt` (or target-specific report names).

## Train / val / test

**60% / 20% / 20%** splits, `random_state=42` (unless overridden), consistent with deviation analysis and metadata capture.

## Staged full-duration forecast

**`4_regression/experiments/combined_duration_forecast.py`** loads (or refits) two stacks of stage models — **primary** with **baseline** features and **post-primary** with **strict_planning** features — then sums components to a **total predicted completion span** and writes a CSV (see script docstring). Persisted bundles live under `6_results/stage_models/` by default or under an experiment directory when using **`main.py --planning-experiment`** or **`4_regression/experiments/planning_experiment_runner.py`**.

## Late-risk classification

**`4_regression/experiments/late_risk_classifier.py`** trains **`HistGradientBoostingClassifier`** (`class_weight='balanced'`) on **strict_planning** features.

**Label (disease-stratified).** `late_risk = 1` iff actual `total_completion` days exceed the training-split Q75 within the trial's **(phase, disease category)** cell. The disease category is the **CCSR domain** (`ccsr_domain` — 21 body-system categories such as `NEO`, `CIR`, `DIG`; trials without a CCSR mapping use `Other_Unclassified`). This follows the principle that a 10-year cardiovascular trial is not "late" — long cardio trials are expected — whereas a 4-year oncology trial may be.

**Hierarchical fallback for sparse cells.** For any cell with fewer than `--min-group-rows` (default **30**) train rows, the classifier falls back to the phase-level Q75, and then to the global train Q75 if the phase itself is also sparse. Every (phase, domain) cell records which rule produced its threshold (`group | phase | global`); the report prints the full table.

**Escape hatch for A/B comparison.** Pass `--disease-axis none` to reproduce the earlier phase-only label (single threshold per phase, same fallback to global for sparse phases). Useful for quantifying the impact of stratification in the final deck.

Outputs:

- `6_results/late_risk_classification_report.txt` — metrics plus the per-group threshold table.
- `6_results/late_risk_predictions.csv` — per-trial columns include `disease_category`, `late_threshold_days`, `late_threshold_source`, `late_risk_true`, `late_risk_pred_proba`, `late_risk_pred`.

Reported metrics (train / val / test): precision, recall, F1, ROC-AUC, PR-AUC, positive rate. **Default decision rule:** `late_risk_pred = 1` when predicted positive-class probability **≥ 0.6** (CLI: `--decision-threshold`).

## Deviation and comparison

- **Deviation:** `5_deviation/deviation_analysis.py` — percent deviation of actual vs predicted for configurable targets; **`--target combined`** consumes the combined forecast CSV.
- - **Risk flagging & accuracy bands:** `5_deviation/baseline_deviation.py` extends the deviation analysis with a ±20% threshold to flag late-risk trials (`late_risk_flag`), and reports accuracy bands (% of trials predicted within ±10%, ±20%, ±30%) for diagnostic interpretation.
- **Baseline vs staged summary:** `4_regression/build_final_comparison_report.py` aggregates regression reports, optional **`baseline_metadata.json`**, optional **frozen** primary regression report, and the late-risk report into **`final_comparison_metrics.csv`** and a Markdown (optional TXT) narrative.

## Feature importance (`4_regression/experiments/feature_importance_ranking.py`)

Permutation-importance report for both models, written to **`6_results/feature_importance_rankings.txt`** (change path via `--output`):

- **Baseline regression** — primary-completion HGBR (TransformedTargetRegressor), `scoring="r2"` on the train fold.
- **Late-risk classifier** — same setup as `late_risk_classifier.py`, `scoring="roc_auc"` on the train fold.

Both runs use `sklearn.inspection.permutation_importance` (recent HGBR/HGBC no longer expose an impurity-based `feature_importances_`), 4 repeats, up to 12k rows sampled, and one-hot columns are reported individually (e.g. `category_NEO`).

## Metrics

Regression: **RMSE**, **MAE**, **R²** on train / val / test as printed in each report. Classification: see late-risk report. For a single-command refresh of the default primary baseline report, run `python 4_regression/core/step03_train_regression.py` and open **`6_results/regression_report.txt`**.

## Latest verified baseline results

From the latest local run of:

- `python 3_preprocessing/run_condition_mapping.py`
- `python 3_preprocessing/preprocess.py`
- `python 4_regression/core/step03_train_regression.py`
- `PYTHONPATH=4_regression python 4_regression/experiments/late_risk_classifier.py`

(Condensed copy: **`final_results_capstone.txt`** at repo root.)

Snapshot:

- Preprocessed filtered trials: **84,879**
- Completed modeling cohort: **57,865**
- Condition-mapping coverage in completed cohort (`has_ccsr=1`): **73.9%**

Primary-completion test metrics (baseline feature policy) — latest run (`final_results_capstone.txt`, 2026-04-20):

| Phase / route | Test n | R² | RMSE (days) | MAE (days) |
|---|---:|---:|---:|---:|
| PHASE1 (dedicated) | 4,293 | 0.5981 | 331 | 183 |
| PHASE2 (dedicated) | 3,305 | 0.4206 | 481 | 323 |
| PHASE3 (dedicated) | 3,093 | 0.4189 | 453 | 301 |
| PHASE1/PHASE2 (early joint routing) | 640 | 0.3571 | 606 | 419 |
| PHASE2/PHASE3 (late joint routing) | 216 | 0.2465 | 576 | 380 |

Late-risk classifier (strict planning features, test split, **probability threshold 0.6**): precision **0.5444**, recall **0.4759**, F1 **0.5079**, ROC-AUC **0.7771**, PR-AUC **0.5741**.

### Top permutation-importance features (latest run)

Normalized to sum to 1 per model (full list in `6_results/feature_importance_rankings.txt`).

| Rank | Baseline regression (R²) | Late-risk classifier (ROC-AUC) |
|:---:|---|---|
| 1 | `max_planned_followup_days` (0.160) | `max_planned_followup_days` (0.208) |
| 2 | `start_year` (0.058) | `category_NEO` (0.116) |
| 3 | `category_NEO` (0.045) | `enrollment` (0.082) |
| 4 | `maximum_age` (0.043) | `maximum_age` (0.055) |
| 5 | `enrollment` (0.034) | `eligibility_n_inclusion_tildes` (0.031) |
