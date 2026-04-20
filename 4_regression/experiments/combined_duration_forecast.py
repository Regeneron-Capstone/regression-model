#!/usr/bin/env python3
"""
Combined full-duration forecast: primary-completion (baseline features) + post-primary (strict planning), summed.

Loads sklearn bundles from ``6_results/stage_models/`` (fits and saves them if missing or with ``--refit``).
Each bundle is one HistGradientBoostingRegressor + TransformedTargetRegressor(log1p/expm1), trained on
all COMPLETED rows in that phase pool after the usual target filter — same routing as ``train_regression``.

Outputs CSV (default ``6_results/combined_duration_predictions.csv``) with predictions, actuals when defined,
and logged sanity checks (no NaNs in predictions, no negatives after clipping).

Usage (repo root):
  python 4_regression/experiments/combined_duration_forecast.py
  python 4_regression/experiments/combined_duration_forecast.py --refit --output 6_results/combined_duration_predictions.csv
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
_REGRESSION_DIR = _SCRIPT_DIR.parent
_CORE_DIR = _REGRESSION_DIR / "core"
for p in (_REGRESSION_DIR, _CORE_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from step01_features import transform_feature_matrix  # noqa: E402
from step02_targets import resolve_target_series  # noqa: E402
from cohort_columns import (  # noqa: E402
    EARLY_JOINT_PHASES,
    KEPT_ARM_INTERVENTION,
    KEPT_DESIGN,
    KEPT_DESIGN_OUTCOMES,
    KEPT_ELIGIBILITY,
    KEPT_ELIGIBILITY_CRITERIA_TEXT,
    KEPT_SITE_FOOTPRINT,
    LATE_JOINT_PHASES,
    default_feature_prep_kw,
)
from step00_cohort_io import load_and_join  # noqa: E402
from step03_train_regression import RESULTS_DIR, _new_regressor, prepare_features  # noqa: E402

logger = logging.getLogger(__name__)

RANDOM_STATE_HGBR = 42

# Same routing as train_regression joint / dedicated models
PHASE_TO_SLOT: dict[str, str] = {
    "PHASE1": "dedicated_PHASE1",
    "PHASE2": "dedicated_PHASE2",
    "PHASE3": "dedicated_PHASE3",
    "PHASE1/PHASE2": "joint_early",
    "PHASE2/PHASE3": "joint_late",
}

ALL_SLOTS: tuple[str, ...] = (
    "dedicated_PHASE1",
    "dedicated_PHASE2",
    "dedicated_PHASE3",
    "joint_early",
    "joint_late",
)

STAGE_CONFIGS: tuple[dict[str, Any], ...] = (
    {
        "tag": "primary_baseline",
        "target_kind": "primary_completion",
        "feature_policy": "baseline",
    },
    {
        "tag": "post_primary_planning",
        "target_kind": "post_primary_completion",
        "feature_policy": "strict_planning",
    },
)


def _base_prep_kw() -> dict[str, Any]:
    d = default_feature_prep_kw(policy="baseline", target_kind="primary_completion")
    return {k: v for k, v in d.items() if k not in ("policy", "target_kind")}


def _prep_kw_for_transform() -> dict[str, Any]:
    """Column lists for ``transform_feature_matrix`` (policy read from artifacts)."""
    return _base_prep_kw()


def cohort_for_slot(completed: pd.DataFrame, slot: str) -> pd.DataFrame:
    if slot.startswith("dedicated_"):
        ph = slot[len("dedicated_") :]
        return completed[completed["phase"] == ph].copy()
    if slot == "joint_early":
        return completed[completed["phase"].isin(EARLY_JOINT_PHASES)].copy()
    if slot == "joint_late":
        return completed[completed["phase"].isin(LATE_JOINT_PHASES)].copy()
    raise ValueError(f"Unknown slot: {slot!r}")


def _bundle_path(models_root: Path, tag: str, slot: str) -> Path:
    return models_root / tag / f"{slot}.joblib"


def fit_and_save_bundles(
    completed: pd.DataFrame,
    models_root: Path,
    *,
    refit: bool,
) -> None:
    """Fit full-cohort models per slot and persist joblib bundles."""
    prep_transform = _prep_kw_for_transform()
    models_root.mkdir(parents=True, exist_ok=True)

    for cfg in STAGE_CONFIGS:
        tag = cfg["tag"]
        target_kind = cfg["target_kind"]
        feature_policy = cfg["feature_policy"]
        out_dir = models_root / tag
        out_dir.mkdir(parents=True, exist_ok=True)

        for slot in ALL_SLOTS:
            path = _bundle_path(models_root, tag, slot)
            if path.exists() and not refit:
                continue

            cohort = cohort_for_slot(completed, slot)
            if len(cohort) < 30:
                logger.warning(
                    "Skip fit %s / %s: cohort only %s rows (need >= 30)",
                    tag,
                    slot,
                    len(cohort),
                )
                if path.exists():
                    path.unlink()
                continue

            fit_kw = {
                **prep_transform,
                "target_kind": target_kind,
                "policy": feature_policy,
            }
            X, y, _, artifacts = prepare_features(cohort, **fit_kw)
            if len(y) < 30:
                logger.warning(
                    "Skip fit %s / %s: after target filter only %s rows",
                    tag,
                    slot,
                    len(y),
                )
                if path.exists():
                    path.unlink()
                continue

            model = _new_regressor()
            model.fit(X, y)

            bundle = {
                "model": model,
                "artifacts": artifacts,
                "prep_kw": prep_transform,
                "slot": slot,
                "target_kind": target_kind,
                "feature_policy": feature_policy,
                "random_state_hgbr": RANDOM_STATE_HGBR,
                "n_train_rows": int(len(y)),
            }
            joblib.dump(bundle, path)
            logger.info(
                "Saved bundle %s rows=%s path=%s",
                f"{tag}/{slot}",
                f"{len(y):,}",
                path,
            )


def load_bundles(models_root: Path, tag: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for slot in ALL_SLOTS:
        path = _bundle_path(models_root, tag, slot)
        if path.exists():
            out[slot] = joblib.load(path)
    return out


def predict_slot_column(
    completed: pd.DataFrame,
    slot_assignments: np.ndarray,
    bundles: dict[str, dict[str, Any]],
    slot: str,
) -> np.ndarray:
    """Predict for all rows where assignment == slot."""
    mask = slot_assignments == slot
    n = len(completed)
    col = np.full(n, np.nan, dtype=np.float64)
    if not np.any(mask) or slot not in bundles:
        return col
    sub = completed.loc[mask].reset_index(drop=True)
    b = bundles[slot]
    X = transform_feature_matrix(sub, b["artifacts"], **b["prep_kw"])
    pred = b["model"].predict(X)
    col[mask] = pred
    return col


def run_forecast(
    completed: pd.DataFrame,
    models_root: Path,
    *,
    refit: bool,
) -> pd.DataFrame:
    completed = completed.copy()
    completed["phase"] = completed["phase"].astype(str)
    completed = completed.reset_index(drop=True)

    fit_and_save_bundles(completed, models_root, refit=refit)

    primary = load_bundles(models_root, "primary_baseline")
    post = load_bundles(models_root, "post_primary_planning")

    logger.info(
        "Loaded primary_baseline bundles: %s / %s slots",
        len(primary),
        len(ALL_SLOTS),
    )
    logger.info(
        "Loaded post_primary_planning bundles: %s / %s slots",
        len(post),
        len(ALL_SLOTS),
    )

    phases = completed["phase"].to_numpy(dtype=object)
    slot_assignments = np.array([PHASE_TO_SLOT.get(str(p), "__missing__") for p in phases], dtype=object)
    missing_mask = slot_assignments == "__missing__"
    if np.any(missing_mask):
        vc = pd.Series(phases[missing_mask]).value_counts()
        logger.warning("Rows with unknown phase label (no model route): %s", vc.head(20).to_dict())

    n = len(completed)
    primary_pred = np.full(n, np.nan, dtype=np.float64)
    post_pred = np.full(n, np.nan, dtype=np.float64)
    for slot in ALL_SLOTS:
        part = predict_slot_column(completed, slot_assignments, primary, slot)
        primary_pred = np.where(np.isfinite(part), part, primary_pred)
    for slot in ALL_SLOTS:
        part = predict_slot_column(completed, slot_assignments, post, slot)
        post_pred = np.where(np.isfinite(part), part, post_pred)

    primary_pred = np.maximum(np.asarray(primary_pred, dtype=np.float64), 0.0)
    post_pred = np.maximum(np.asarray(post_pred, dtype=np.float64), 0.0)
    total_pred = primary_pred + post_pred

    act_pri = resolve_target_series(completed, "primary_completion").astype("float64").to_numpy()
    act_post = resolve_target_series(completed, "post_primary_completion").astype("float64").to_numpy()
    act_tot = resolve_target_series(completed, "total_completion").astype("float64").to_numpy()

    out = pd.DataFrame(
        {
            "nct_id": completed["nct_id"].values,
            "phase": completed["phase"].values,
            "predicted_primary_completion_days": primary_pred,
            "predicted_post_primary_completion_days": post_pred,
            "predicted_total_completion_days": total_pred,
            "actual_primary_completion_days": act_pri,
            "actual_post_primary_completion_days": act_post,
            "actual_total_completion_days": act_tot,
        }
    )
    return out


def sanity_check_predictions(df: pd.DataFrame) -> None:
    req = (
        "predicted_primary_completion_days",
        "predicted_post_primary_completion_days",
        "predicted_total_completion_days",
    )
    for c in req:
        if df[c].isna().any():
            n_bad = int(df[c].isna().sum())
            raise ValueError(f"Sanity check failed: {n_bad} missing values in {c!r}")
        if (df[c] < 0).any():
            n_bad = int((df[c] < 0).sum())
            raise ValueError(f"Sanity check failed: {n_bad} negative values in {c!r}")
    logger.info(
        "Sanity OK: all finite non-negative predictions (n=%s)",
        f"{len(df):,}",
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Combined primary + post-primary duration forecast CSV.")
    p.add_argument(
        "--models-dir",
        type=Path,
        default=RESULTS_DIR / "stage_models",
        help="Directory for per-slot joblib bundles (default: 6_results/stage_models).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR / "combined_duration_predictions.csv",
        help="Output CSV path.",
    )
    p.add_argument(
        "--refit",
        action="store_true",
        help="Re-fit all stage models and overwrite saved bundles.",
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _parse_args()
    models_root = args.models_dir.expanduser().resolve()
    out_path = args.output.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Combined duration forecast: models_dir=%s output=%s refit=%s hgbr_random_state=%s",
        models_root,
        out_path,
        args.refit,
        RANDOM_STATE_HGBR,
    )

    df = load_and_join(
        eligibility_columns=KEPT_ELIGIBILITY,
        site_footprint_columns=KEPT_SITE_FOOTPRINT,
        design_columns=KEPT_DESIGN,
        arm_intervention_columns=KEPT_ARM_INTERVENTION,
        design_outcomes_columns=KEPT_DESIGN_OUTCOMES,
    )
    logger.info("Loaded cohort: %s COMPLETED trials", f"{len(df):,}")

    result = run_forecast(df, models_root, refit=args.refit)
    sanity_check_predictions(result)
    result.to_csv(out_path, index=False)
    logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
