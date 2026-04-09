# Regeneron Capstone — Clinical trial duration model

Industry-sponsored clinical trials from ClinicalTrials.gov, preprocessed into `clean_data/`, then modeled with **scikit-learn** for **COMPLETED** trials. The default regression target is **primary completion span** (`duration_days` / start → primary completion); additional targets, feature policies, and side models are documented in **`MODEL.md`**.

**Core regression** (`4_regression/train_regression.py`):

- **Dedicated models** on **PHASE1**, **PHASE2**, **PHASE3** only.
- **Early joint** pool **PHASE1 + PHASE1/PHASE2 + PHASE2** → scores **PHASE1/PHASE2** rows.
- **Late joint** pool **PHASE2 + PHASE2/PHASE3 + PHASE3** → scores **PHASE2/PHASE3** rows.

Metrics are reported **per phase label** (`PHASE_REPORT_ORDER` in `4_regression/cohort_columns.py`). Shared column bundles and data loading live in **`4_regression/cohort_columns.py`** and **`4_regression/cohort_io.py`** (not tied to a single training script).

High-level modeling rules: **`MODEL.md`**. Architecture and pipeline order: **`CODEBASE.md`**.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
```

Place raw extracts under **`raw_data/`** (e.g. from `1_scripts/` BigQuery downloads), then:

```bash
python 3_preprocessing/preprocess.py                 # builds clean_data/
python 4_regression/train_regression.py              # results/regression_report.txt (primary + baseline features)
```

**Full pipeline** (downloads, EDA, preprocess, train):

```bash
python main.py
python main.py --skip-download    # when raw_data/ is already populated
```

**More regression / planning tracks** (after preprocess; many need `PYTHONPATH=4_regression` when run from repo root):

```bash
PYTHONPATH=4_regression python 4_regression/train_regression.py --target post_primary_completion --feature-policy strict_planning
PYTHONPATH=4_regression python 4_regression/train_post_primary_planning.py   # shortcut for the line above
PYTHONPATH=4_regression python 4_regression/combined_duration_forecast.py    # primary + post-primary stage sum → CSV
PYTHONPATH=4_regression python 4_regression/late_risk_classifier.py           # late-risk classification (strict features)
PYTHONPATH=4_regression python 4_regression/capture_baseline_metadata.py       # optional baseline_metadata.json
```

**Deviation analysis** (actual vs predicted %):

```bash
PYTHONPATH=4_regression python 5_deviation/deviation_analysis.py --target primary_completion
PYTHONPATH=4_regression python 5_deviation/deviation_analysis.py --target combined --splits test   # needs combined_duration_predictions.csv
# Legacy wrapper: python 5_deviation/baseline_deviation.py
```

**End-to-end planning experiment** (train primary → train post-primary strict → combined forecast → late-risk → combined deviation); artifacts under `results/experiments/<UTC_timestamp>/`:

```bash
python main.py --skip-download --planning-experiment    # after explore + preprocess; add --experiment-dry-run to print commands only
python 4_regression/planning_experiment_runner.py       # same staged steps without main’s download/EDA/preprocess
python 4_regression/planning_experiment_runner.py --dry-run
```

**Baseline vs staged comparison report** (aggregates reports + optional frozen baseline file):

```bash
PYTHONPATH=4_regression python 4_regression/build_final_comparison_report.py \
  --metadata results/baseline_metadata.json \
  --frozen-regression results/frozen/regression_report_baseline_primary.txt \
  --out-csv results/final_comparison_metrics.csv \
  --out-md results/final_comparison_report.md \
  --out-txt results/final_comparison_report.txt
```

---

## Results

After training, open **`results/regression_report.txt`** for train / val / test **RMSE**, **MAE**, and **R²** by model block (dedicated + joint), plus summary tables. Other runs write named files under **`results/`** (see `train_regression.resolve_report_path` and each script’s defaults). Generated outputs are gitignored under **`results/`** by default.

**Design choices:** no `StandardScaler` on features; missing numerics stay **NaN** for HGBR; target transform **log1p** / **expm1** inside `TransformedTargetRegressor`; **`max_iter=200`** on boosters. Example headline test R² from a prior primary run: Phase 1 **≈ 0.60**; Phases 2–3 **≈ 0.42–0.43** — re-run to refresh after data or code changes.

---

## Repository layout (high level)

| Path | Role |
|------|------|
| `1_scripts/` | BigQuery download helpers → `raw_data/` |
| `2_data_exploration/` | EDA; `run_all.py` |
| `3_preprocessing/` | `preprocess.py` → `clean_data/` |
| `4_regression/` | `train_regression.py`, `planning_experiment_runner.py`, `cohort_io.py`, …; combined forecast, late-risk, comparison report builder, etc. |
| `5_deviation/` | `deviation_analysis.py`, `baseline_deviation.py` |
| `main.py` | Full pipeline; add `--planning-experiment` for the staged experiment instead of a single baseline train |

Raw data: **`raw_data/`**. Builds: **`clean_data/`**, **`results/`** (see **`.gitignore`**).
