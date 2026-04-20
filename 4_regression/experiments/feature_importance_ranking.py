#!/usr/bin/env python3
"""
Rank features via ``sklearn.inspection.permutation_importance`` (means renormalized to sum to 1) for:

1. **Baseline regression** — primary completion duration (``HistGradientBoostingRegressor`` inside
   ``TransformedTargetRegressor`` with log1p/expm1), scoring **R²** on the train fold.
2. **Late-risk classifier** — same label construction as ``late_risk_classifier.py`` (strict_planning features,
   hierarchical Q75 on total completion), scoring **ROC-AUC** on the train fold.

Usage (repo root):

  PYTHONPATH=4_regression python 4_regression/experiments/feature_importance_ranking.py
  PYTHONPATH=4_regression python 4_regression/experiments/feature_importance_ranking.py --output feature_importance_rankings.txt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

_SCRIPT_DIR = Path(__file__).resolve().parent
_REGRESSION_DIR = _SCRIPT_DIR.parent
_CORE_DIR = _REGRESSION_DIR / "core"
for p in (_REGRESSION_DIR, _CORE_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from cohort_columns import (  # noqa: E402
    KEPT_ARM_INTERVENTION,
    KEPT_DESIGN,
    KEPT_DESIGN_OUTCOMES,
    KEPT_ELIGIBILITY,
    KEPT_SITE_FOOTPRINT,
    default_feature_prep_kw,
)
from late_risk_classifier import (  # noqa: E402
    DiseaseAxis,
    _align_domains,
    _apply_threshold_map,
    _fit_threshold_map,
    _prep_kw_strict,
)
from step00_cohort_io import load_and_join  # noqa: E402
from step01_features import feature_matrix_column_names  # noqa: E402
from step03_train_regression import _new_regressor, _train_val_test_split, prepare_features  # noqa: E402

PROJECT_ROOT = _REGRESSION_DIR.parent
DEFAULT_OUT = PROJECT_ROOT / "feature_importance_rankings.txt"

_PERM_REPEATS = 4
_PERM_MAX_SAMPLES = 12_000


def _permutation_importances(
    estimator,
    X: np.ndarray,
    y: np.ndarray,
    *,
    random_state: int,
    scoring: str,
) -> np.ndarray:
    """Mean permutation importance; normalized to sum to 1 for readability."""
    n_samples = min(_PERM_MAX_SAMPLES, len(X))
    r = permutation_importance(
        estimator,
        X,
        y,
        n_repeats=_PERM_REPEATS,
        random_state=random_state,
        scoring=scoring,
        max_samples=n_samples,
        n_jobs=-1,
    )
    imp = np.maximum(np.asarray(r.importances_mean, dtype=float), 0.0)
    s = float(imp.sum())
    if s > 0:
        imp = imp / s
    return imp


def _rank_lines(title: str, names: list[str], importances: np.ndarray) -> list[str]:
    if len(names) != len(importances):
        raise ValueError(f"{title}: names={len(names)} importances={len(importances)}")
    order = np.argsort(-importances)
    lines = [title, "=" * len(title), ""]
    lines.append(f"Features: {len(names):,}  |  scores sum to {importances.sum():.6f} (normalized)")
    lines.append("")
    for rank, j in enumerate(order, start=1):
        lines.append(f"{rank:4d}.  {importances[j]:.6f}  {names[j]}")
    lines.append("")
    return lines


def run_regression_ranking(cohort: pd.DataFrame, random_state: int) -> list[str]:
    prep = default_feature_prep_kw(policy="baseline", target_kind="primary_completion")
    X, y, _, art = prepare_features(cohort, **prep)
    names = feature_matrix_column_names(art)
    X_train, _, _, y_train, _, _ = _train_val_test_split(X, y, random_state=random_state)
    model = _new_regressor()
    model.fit(X_train, y_train)
    imp = _permutation_importances(
        model, X_train, y_train, random_state=random_state, scoring="r2"
    )
    meta = [
        "MODEL 1 — Primary completion duration (baseline features)",
        "-" * 72,
        f"Estimator: HistGradientBoostingRegressor in TransformedTargetRegressor (log1p / expm1)",
        f"Importance: permutation_importance on train fold (scoring=r2, "
        f"n_repeats={_PERM_REPEATS}, max_samples=min({_PERM_MAX_SAMPLES:,}, n_train))",
        f"Rows (after target filter): {len(y):,}",
        f"Train rows (60% of matrix rows): {len(y_train):,}",
        "Encoders are fit on the full filtered cohort before splitting (same as step03_train_regression).",
        "",
    ]
    return meta + _rank_lines("Rank (all features)", names, imp)


def run_classifier_ranking(
    cohort: pd.DataFrame,
    *,
    random_state: int,
    late_quantile: float,
    min_group_rows: int,
    disease_axis: DiseaseAxis,
) -> list[str]:
    df = cohort.copy()
    df["phase"] = df["phase"].astype(str)
    prep = _prep_kw_strict()
    X, y_cont, phases, art = prepare_features(df, **prep)
    nct_ids = art.get("nct_ids")
    if nct_ids is None:
        raise RuntimeError("Expected nct_ids in feature artifacts")
    domains = _align_domains(df, nct_ids, disease_axis)

    idx = np.arange(len(y_cont))
    i_train, _ = train_test_split(idx, test_size=0.4, random_state=random_state, shuffle=True)

    tmap = _fit_threshold_map(
        y_cont[i_train],
        phases[i_train],
        domains[i_train],
        quantile=late_quantile,
        min_group_rows=min_group_rows,
        disease_axis=disease_axis,
    )
    y_train, _, _ = _apply_threshold_map(y_cont[i_train], phases[i_train], domains[i_train], tmap)

    clf = HistGradientBoostingClassifier(
        max_iter=200,
        random_state=random_state,
        class_weight="balanced",
    )
    clf.fit(X[i_train], y_train)
    imp = _permutation_importances(
        clf,
        X[i_train],
        y_train,
        random_state=random_state,
        scoring="roc_auc",
    )
    names = feature_matrix_column_names(art)

    meta = [
        "MODEL 2 — Late-risk classification (strict_planning features)",
        "-" * 72,
        "Estimator: HistGradientBoostingClassifier (class_weight=balanced)",
        f"Importance: permutation_importance on train fold (scoring=roc_auc, "
        f"n_repeats={_PERM_REPEATS}, max_samples=min({_PERM_MAX_SAMPLES:,}, n_train))",
        f"Rows (total_completion strict matrix): {len(y_cont):,}",
        f"Train rows: {len(i_train):,}",
        f"Binary label: actual total_completion_days > Q{late_quantile:g} within (phase, disease domain); "
        f"disease_axis={disease_axis!r}; min_group_rows={min_group_rows}",
        "",
    ]
    return meta + _rank_lines("Rank (all features)", names, imp)


def main() -> None:
    p = argparse.ArgumentParser(description="Feature importance: baseline regression + late-risk classifier")
    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output report (default: {DEFAULT_OUT})",
    )
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--late-quantile", type=float, default=0.75)
    p.add_argument("--min-group-rows", type=int, default=30)
    p.add_argument("--disease-axis", choices=("ccsr_domain", "none"), default="ccsr_domain")
    args = p.parse_args()
    if not (0.0 < args.late_quantile < 1.0):
        raise SystemExit("--late-quantile must be in (0, 1)")
    if args.min_group_rows < 1:
        raise SystemExit("--min-group-rows must be >= 1")

    cohort = load_and_join(
        eligibility_columns=KEPT_ELIGIBILITY,
        site_footprint_columns=KEPT_SITE_FOOTPRINT,
        design_columns=KEPT_DESIGN,
        arm_intervention_columns=KEPT_ARM_INTERVENTION,
        design_outcomes_columns=KEPT_DESIGN_OUTCOMES,
    )

    out_lines: list[str] = [
        "FEATURE IMPORTANCE RANKINGS",
        "=" * 72,
        "Importances = permutation_importance means (non-negative), renormalized to sum to 1 per model.",
        "HistGradientBoosting* in current sklearn does not expose impurity-based feature_importances_.",
        "One-hot columns are separate features (e.g. category_NEO).",
        "",
    ]

    out_lines.extend(run_regression_ranking(cohort, args.random_state))
    out_lines.extend(
        run_classifier_ranking(
            cohort,
            random_state=args.random_state,
            late_quantile=args.late_quantile,
            min_group_rows=args.min_group_rows,
            disease_axis=args.disease_axis,
        )
    )

    out_path = args.output.expanduser().resolve()
    out_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
