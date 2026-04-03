# Duration Regression Model

`sklearn.ensemble.HistGradientBoostingRegressor` predicting trial duration (days) for **COMPLETED** trials only — **one model per phase label** (PHASE1, PHASE1/PHASE2, PHASE2, PHASE2/PHASE3, PHASE3), no global scaling; the booster uses non-linear splits and native **NaN** handling in numeric features.

## Target
- `duration_days` — time from start to primary completion
- Preprocessing keeps only trials with **14 ≤ duration_days ≤ 3650** (drop sub-two-week and over-10-year windows as outliers)

## Features (ablation-tested, best-performing subset)

### Core features (always included)
- `phase` — defines which dedicated model is trained (not one-hot encoded inside each single-phase model)
- `enrollment` — planned enrollment
- `n_sponsors` — number of sponsors
- `number_of_arms` — number of arms
- `start_year` — trial start year
- `category` — therapeutic category (one-hot, 132 levels)
- `downcase_mesh_term` — MeSH condition terms (one-hot)
- `intervention_type` — intervention types (one-hot)

### Eligibility (kept from ablation)
- `gender`, `minimum_age`, `maximum_age`, `adult`, `child`, `older_adult`

### Site footprint (kept from ablation)
- `number_of_facilities`, `number_of_countries`, `us_only`, `has_single_facility`

### Design (kept from ablation)
- `randomized`, `intervention_model`, `masking_depth_score`, `primary_purpose`, `design_complexity_composite`

### Arm/intervention (kept from ablation)
- `number_of_interventions`, `intervention_type_diversity`, `mono_therapy`, `has_placebo`, `has_active_comparator`, `n_mesh_intervention_terms`

### Design outcomes (from design_outcomes table)
- `max_planned_followup_days` — max planned follow-up parsed from time_frame
- `n_primary_outcomes`, `n_secondary_outcomes`, `n_outcomes`
- `has_survival_endpoint`, `has_safety_endpoint` — flags from measure/description
- `endpoint_complexity_score` — composite of outcome count and endpoint types

## Training
- Five independent `HistGradientBoostingRegressor` models, one per label: **PHASE1**, **PHASE1/PHASE2**, **PHASE2**, **PHASE2/PHASE3**, **PHASE3**.
- **No** `StandardScaler`; numeric features keep **NaN** where missing so HGBR can use missingness in splits.
- Phase is **not** one-hot encoded inside each model (constant within cohort).

## Metrics
- See `results/regression_report.txt` after `python 4_regression/train_regression.py` — train / val / **test** RMSE, MAE, R² **per phase**, plus a summary of test R² by phase.

## Train/val/test split
Per phase: 60% / 20% / 20%, `random_state=42`
