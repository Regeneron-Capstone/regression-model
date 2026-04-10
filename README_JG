# Clinical Trial Duration Prediction

A machine learning pipeline for predicting clinical trial completion timelines
using regression models trained on completed, industry-sponsored trials from
ClinicalTrials.gov.

---

## Overview

This project forecasts clinical trial duration — specifically, the number of
days from trial start date to primary completion date — using gradient boosting
regression. Phase-specific models are trained independently and evaluated
against held-out test sets. A secondary late-risk classifier identifies trials
at elevated risk of exceeding expected duration.

---

## Problem Statement

Clinical trial planning requires accurate duration forecasts at the time of
study design, before enrollment and execution data are available. This pipeline
produces three duration estimates per trial:

1. **Primary completion** — start date to primary completion date
2. **Post-primary completion** — primary completion date to study completion date
3. **Total completion** — start date to final study completion date

---

## Data Source

Data is sourced from Google BigQuery:

- **Project**: `regeneron-capstone-delta`
- **Dataset**: `regeneron_capstone_delta_dataset`

The following tables are downloaded:

| Table | Contents |
|---|---|
| `studies` | Core trial metadata |
| `sponsors` | Sponsor information |
| `eligibilities` | Eligibility criteria |
| `browse_conditions` | MeSH condition terms |
| `browse_interventions` | Intervention MeSH terms |
| `facilities` | Trial site locations |
| `countries` | Country data |
| `design_groups` | Trial arm definitions |
| `design_outcomes` | Outcome definitions |
| `designs` | Study design parameters |
| `interventions` | Intervention details |
| `calculated_values` | Pre-computed derived fields |

Raw downloads are stored in `raw_data/` as CSV or Parquet files.

---

## Cohort Definition

The analysis is restricted to trials meeting all of the following criteria:

- **Sponsor type**: Industry-sponsored only (`agency_class == 'INDUSTRY'`)
- **Study type**: Interventional
- **Status**: Completed (withdrawn trials excluded)
- **Phases**: PHASE1, PHASE2, PHASE3, PHASE1/PHASE2, PHASE2/PHASE3
- **Date range**: Start dates between 1980 and 2027
- **Duration band**: 14 to 3,650 days (10 years)

---

## Pipeline

The pipeline is organized into five numbered stages:

```
1_scripts/            Download raw tables from BigQuery
2_data_exploration/   Exploratory data analysis
3_preprocessing/      Data cleaning and feature engineering
4_regression/         Model training, evaluation, and forecasting
5_deviation/          Prediction error analysis
```

### Stage 1 — Data Download (`1_scripts/`)

Each BigQuery table has a dedicated download script (e.g.,
`download_studies.py`, `download_sponsors.py`). The core downloader
(`bq_downloader.py`) supports incremental fetching via row-count checkpoints
stored in JSON, and a force-download option to bypass caching.

### Stage 2 — Exploration (`2_data_exploration/`)

Run all EDA scripts via:

```bash
python 2_data_exploration/run_all.py
```

Individual scripts cover: study metadata, sponsor distributions, MeSH term
frequencies, intervention types, eligibility criteria, site footprints, study design patterns, and planned follow-up durations.

### Stage 3 — Preprocessing (`3_preprocessing/`)

`preprocess.py` applies the cohort filters above, computes duration targets,
merges eligibility features, and outputs:

- `clean_data/studies.csv`
- `clean_data/sponsors.csv`
- `clean_data/enrollment_stats_by_phase.csv`
- `clean_data/preprocessing_summary.txt`

`sanity_check.py` reports descriptive statistics on duration distributions
(min, max, mean, median, standard deviation, percentiles).

### Stage 4 — Regression (`4_regression/`)

See [Modeling](#modeling) for details on algorithms and training strategy.

Key scripts:

| Script | Purpose |
|---|---|
| `train_regression.py` | Primary completion model (baseline features) |
| `train_post_primary_planning.py` | Post-primary completion model (planning features) |
| `combined_duration_forecast.py` | Two-stage total duration forecast |
| `late_risk_classifier.py` | Binary late-risk classification |
| `planning_experiment_runner.py` | Full five-stage experiment orchestrator |
| `deviation_analysis.py` | Prediction error and accuracy-band analysis |
| `build_final_comparison_report.py` | Baseline vs. staged model comparison |
| `compare_feature_policies.py` | Baseline vs. strict-planning feature comparison |

### Stage 5 — Deviation Analysis (`5_deviation/`)

Standalone deviation analysis script. Also executed automatically as stage 5
of `planning_experiment_runner.py`.

---

## Modeling

### Algorithm

**Regression**: `HistGradientBoostingRegressor` wrapped in `TransformedTargetRegressor`

- Target transformation: `log1p` (forward) / `expm1` (inverse)
- `max_iter=200`, `random_state=42`
- Missing numeric values are not imputed; the algorithm handles them natively
- No feature standardization applied

**Classification (late-risk)**: `HistGradientBoostingClassifier`

- `max_iter=200`, `random_state=42`, `class_weight="balanced"`
- Label: `late_risk = 1` if actual total days > Q75 within phase
  (falls back to global training quantile for phases with fewer than 30 samples)

### Training Strategy

Models are trained separately per phase to account for structural differences
in trial design and duration. Two additional joint strategies pool adjacent
phases to improve coverage for mixed-phase trials:

| Strategy | Phases Pooled |
|---|---|
| Dedicated | PHASE1, PHASE2, PHASE3 (independent) |
| Early joint | PHASE1, PHASE1/PHASE2, PHASE2 |
| Late joint | PHASE2, PHASE2/PHASE3, PHASE3 |

**Data split**: 60% train / 20% validation / 20% test (stratified by phase,
`random_state=42`)

### Feature Policies

Two feature policies control which inputs are available at prediction time:

- **`baseline`**: All engineered features, including `start_year` and
  site-footprint fields (number of facilities, countries, US-only flag, etc.)
- **`strict_planning`**: Excludes `start_year` and all site-footprint fields,
  retaining only information available at the planning stage before sites are
  selected

### Features

| Group | Features |
|---|---|
| Trial | `phase`, `enrollment`, `n_sponsors`, `number_of_arms`, `start_year`\*, `category` |
| Condition | Top-50 MeSH terms (one-hot), remainder as "other" |
| Intervention | `intervention_type` (top 15), `number_of_interventions`, `intervention_type_diversity`, `mono_therapy`, `has_placebo`, `has_active_comparator`, `n_mesh_intervention_terms` |
| Eligibility (structured) | `gender`, `minimum_age`, `maximum_age`, `adult`, `child`, `older_adult` |
| Eligibility (text) | Criteria text length, inclusion count, exclusion count, burden procedure flag (biopsy / MRI / endoscopy / PET scan) |
| Site footprint\* | `number_of_facilities`, `number_of_countries`, `us_only`, `has_single_facility`, `number_of_us_states`, `facility_density` |
| Study design | `randomized`, `intervention_model` (top 6), `masking_depth_score`, `primary_purpose` (top 6), `design_complexity_composite` |
| Outcomes | `max_planned_followup_days`, `n_primary_outcomes`, `n_secondary_outcomes`, `n_outcomes`, `has_survival_endpoint`, `has_safety_endpoint`, `endpoint_complexity_score` |

\* Excluded from `strict_planning` policy

---

## Evaluation

### Regression Metrics

| Metric | Description |
|---|---|
| RMSE | Root mean squared error (days) |
| MAE | Mean absolute error (days) |
| R² | Coefficient of determination |

### Reported Performance (Test Set)

| Phase | R² |
|---|---|
| Phase 1 | ~0.60 |
| Phase 2 | ~0.42–0.43 |
| Phase 3 | ~0.42–0.43 |

### Deviation Metrics

Percent deviation is computed as:

```
deviation (%) = (actual − predicted) / (predicted + ε) × 100
```

where ε = 1×10⁻¹⁰.

Reported statistics include MAPE, deviation distribution percentiles (P25, P50,
P75, P90), and accuracy bands: percentage of predictions within ±10%, ±20%,
and ±30% of actual. A trial is flagged as "late" if its percent deviation
exceeds 20% (default threshold).

### Classification Metrics

Precision, recall, F1, ROC-AUC, PR-AUC, and positive rate are reported for
the late-risk classifier across train, validation, and test splits.

---

## Related Work

This project builds on the benchmark established by **TrialBench**:

> Jintai Chen et al. "TrialBench: Multi-Modal AI-Ready Datasets for Clinical
> Trial Prediction." *Scientific Data* 12, 1564 (2025).
> https://doi.org/10.1038/s41597-025-05680-8

TrialBench ([ML2Health/ML2ClinicalTrials](https://github.com/ML2Health/ML2ClinicalTrials))
provides curated, AI-ready datasets for eight clinical trial prediction tasks
sourced from ClinicalTrials.gov, DrugBank, TrialTrove, and ICD-10. For
duration prediction, their benchmark employs a multi-modal deep learning
architecture combining:

- Message-passing neural networks (MPNNs) for drug molecular structure
- Bio-BERT for eligibility criteria and trial description text
- GRAM (Graph-based Attention Model) for ICD-10 disease hierarchies
- MeSH embedding layers
- DANet blocks for tabular features

**Performance comparison** (R², test set):

| Phase | TrialBench (Chen et al., 2025) | This project |
|---|---|---|
| Phase 1 | 0.6514 ± 0.0085 | ~0.60 |
| Phase 2 | 0.4125 ± 0.0081 | ~0.42–0.43 |
| Phase 3 | 0.3148 ± 0.0085 | ~0.42–0.43 |

Despite using only tabular features and a single gradient boosting algorithm
(no molecular graphs, no pre-trained language models, no disease hierarchies),
this project matches or exceeds TrialBench on Phase 2 and substantially
surpasses it on Phase 3. The Phase 3 gap (~+0.11 R²) is the most significant
finding: a simpler, more interpretable model trained on structured planning-time
features outperforms a multi-modal deep learning baseline for the most complex
and longest trials.

This comparison informed two design decisions:

1. **Phase-specific modeling.** TrialBench's per-phase results revealed that
   Phase 3 is the hardest to generalize across — motivating dedicated models
   and joint pooling strategies rather than a single global model.

2. **Planning-time feature discipline.** TrialBench uses features available
   only post-hoc (e.g., actual site counts, molecular assay outcomes). The
   `strict_planning` feature policy in this project deliberately excludes such
   features to produce estimates that are useful at the trial design stage,
   before execution data exist.

---

## Installation

```bash
pip install -r requirements.txt
```

**Dependencies:**

```
google-cloud-bigquery >= 3.0.0
db-dtypes             >= 1.0.0
pandas                >= 2.0.0
pyarrow               >= 14.0.0
tqdm                  >= 4.65.0
matplotlib            >= 3.7.0
seaborn               >= 0.12.0
scikit-learn          >= 1.3.0
```

---

## Usage

### Full pipeline (with data download)

```bash
python main.py
```

### Full pipeline (skip download, use cached data)

```bash
python main.py --skip-download
```

### Preprocessing only

```bash
python 3_preprocessing/preprocess.py
```

### Regression training only

```bash
python 4_regression/train_regression.py
```

### Planning-time experiment (all five stages)

```bash
PYTHONPATH=4_regression python 4_regression/planning_experiment_runner.py
```

### Dry run (validate pipeline without training)

```bash
PYTHONPATH=4_regression python 4_regression/planning_experiment_runner.py --dry-run
```

### Post-primary and combined forecasts

```bash
PYTHONPATH=4_regression python 4_regression/train_post_primary_planning.py
PYTHONPATH=4_regression python 4_regression/combined_duration_forecast.py
```

### Late-risk classification

```bash
PYTHONPATH=4_regression python 4_regression/late_risk_classifier.py
```

### Deviation analysis

```bash
python 5_deviation/deviation_analysis.py --target primary_completion
```

### Feature policy comparison

```bash
python 4_regression/compare_feature_policies.py
```

### Validation tests

```bash
python tests/validate_feature_registry.py
python tests/validate_targets.py
```

---

## Outputs

| Path | Contents |
|---|---|
| `clean_data/studies.csv` | Preprocessed trial records |
| `clean_data/preprocessing_summary.txt` | Filtering summary |
| `clean_data/enrollment_stats_by_phase.csv` | Enrollment statistics by phase |
| `results/regression_report.txt` | Primary completion model results |
| `results/regression_report_post_primary_completion_strict_planning.txt` | Post-primary model results |
| `results/experiments/<UTC_timestamp>/` | Full experiment output directory |
| `results/experiments/<UTC_timestamp>/predictions.csv` | Per-trial predictions |
| `results/experiments/<UTC_timestamp>/deviation_summary.txt` | Deviation analysis report |

---

## Repository Structure

```
regression-model/
├── main.py
├── requirements.txt
├── 1_scripts/               # BigQuery download scripts
├── 2_data_exploration/      # EDA scripts and outputs
├── 3_preprocessing/         # Preprocessing and sanity checks
├── 4_regression/            # Models, features, evaluation
├── 5_deviation/             # Standalone deviation analysis
├── tests/                   # Feature registry and target validation
├── raw_data/                # Downloaded source tables (gitignored)
├── clean_data/              # Preprocessed outputs (gitignored)
└── results/                 # Model reports and experiment artifacts
```
