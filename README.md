# Regeneron Capstone — Clinical trial duration model

Industry-sponsored clinical trials from ClinicalTrials.gov, preprocessed into `clean_data/`, then modeled with **five** dedicated gradient-boosting models (one per phase label: PHASE1, PHASE1/PHASE2, PHASE2, PHASE2/PHASE3, PHASE3) to predict **`duration_days`** (primary completion minus start) for **completed** trials only.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
python 3_preprocessing/preprocess.py                 # builds clean_data/
python 4_regression/train_regression.py              # writes results/regression_report.txt
```

See **`MODEL.md`** for features, preprocessing rules (including the 14–3650 day duration band), and training details.

---

## Results

After `python 4_regression/train_regression.py`, open **`results/regression_report.txt`** for train / val / test metrics and a **test R² summary** for all five cohorts.

**Design choices that help:** no feature scaling, **missing numerics left as NaN** for the histogram gradient booster, and **separate models** so each phase label gets its own 200-tree ensemble. Example numbers from a prior run (single-phase-only script): Phase 1 test **R² ≈ 0.60**; Phase 2 and 3 were **≈ 0.42–0.43**. Re-run training to refresh metrics after any data or code change.

---

## Repository layout (high level)

| Path | Role |
|------|------|
| `1_scripts/` | BigQuery / download helpers |
| `2_data_exploration/` | EDA scripts |
| `3_preprocessing/` | `preprocess.py` → `clean_data/` |
| `4_regression/` | `train_regression.py` |
| `main.py` | Optional pipeline runner |

Raw CSVs live under **`raw_data/`** (not committed; use your own extracts). Generated artifacts use **`clean_data/`**, **`results/`**, and paths listed in `.gitignore`.
