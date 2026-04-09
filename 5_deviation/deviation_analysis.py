#!/usr/bin/env python3
"""
Prediction deviation analysis for duration models across multiple targets.

Supports:
  - primary_completion   (baseline features; dedicated PHASE1/2/3, test split only by default)
  - post_primary_completion (strict_planning features)
  - total_completion     (baseline features)
  - combined             (reads ``combined_duration_predictions.csv`` — staged sum + components)

Uses ``targets`` for percent deviation / late flags and ``evaluation.format_deviation_summary_report``
for the text report. Data load uses ``cohort_io`` / ``cohort_columns``; feature matrices use
``train_regression.prepare_features`` only (no training-loop imports).

Usage:
  python 5_deviation/deviation_analysis.py --target primary_completion
  python 5_deviation/deviation_analysis.py --target combined --combined-csv results/combined_duration_predictions.csv
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_REGRESSION_DIR = PROJECT_ROOT / "4_regression"
if str(_REGRESSION_DIR) not in sys.path:
    sys.path.insert(0, str(_REGRESSION_DIR))

from cohort_columns import (  # noqa: E402
    KEPT_ARM_INTERVENTION,
    KEPT_DESIGN,
    KEPT_DESIGN_OUTCOMES,
    KEPT_ELIGIBILITY,
    KEPT_SITE_FOOTPRINT,
    PHASE_REPORT_ORDER,
    PHASE_SINGLE_MODELS,
    default_feature_prep_kw,
)
from cohort_io import load_and_join  # noqa: E402
from evaluation import format_deviation_summary_report  # noqa: E402
from targets import calculate_pct_deviation, describe_target_kind, make_late_flag  # noqa: E402
from train_regression import prepare_features  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)

TargetMode = Literal[
    "primary_completion",
    "post_primary_completion",
    "total_completion",
    "combined",
]

# (feature_policy, target_kind for prepare_features)
REGRESSION_TARGET_CONFIG: dict[str, tuple[str, str]] = {
    "primary_completion": ("baseline", "primary_completion"),
    "post_primary_completion": ("strict_planning", "post_primary_completion"),
    "total_completion": ("baseline", "total_completion"),
}

COMBINED_SPECS: tuple[tuple[str, str, str], ...] = (
    ("total_completion", "actual_total_completion_days", "predicted_total_completion_days"),
    ("primary_completion", "actual_primary_completion_days", "predicted_primary_completion_days"),
    (
        "post_primary_completion",
        "actual_post_primary_completion_days",
        "predicted_post_primary_completion_days",
    ),
)


def _category_map(df: pd.DataFrame) -> dict[str, str]:
    out: dict[str, str] = {}
    if "nct_id" not in df.columns or "category" not in df.columns:
        return out
    for nid, cat in zip(df["nct_id"].astype(str), df["category"]):
        out[str(nid)] = str(cat) if pd.notna(cat) else ""
    return out


def train_dedicated_phase_models(
    df: pd.DataFrame,
    *,
    target_mode: str,
    random_state: int = 42,
) -> dict[str, dict[str, Any]]:
    """
    One TransformedTargetRegressor per dedicated phase (PHASE1/2/3), same 60/20/20 split
    as historical baseline_deviation.
    """
    policy, target_kind = REGRESSION_TARGET_CONFIG[target_mode]
    prep_kw = default_feature_prep_kw(policy=policy, target_kind=target_kind)
    results: dict[str, dict[str, Any]] = {}

    for phase in PHASE_SINGLE_MODELS:
        df_p = df[df["phase"].astype(str) == phase].copy()

        if len(df_p) < 30:
            logger.info("Skipping %s — too few rows (%d)", phase, len(df_p))
            continue

        X, y, _, art = prepare_features(df_p, **prep_kw)
        nct = art.get("nct_ids")
        if nct is None or len(nct) != len(y):
            raise RuntimeError("prepare_features must return nct_ids aligned with y (see features.assemble_feature_matrix)")

        if len(y) < 30:
            logger.info("Skipping %s — too few rows after target filter (%d)", phase, len(y))
            continue

        idx = np.arange(len(y))
        X_train, X_temp, y_train, y_temp, i_tr, i_temp = train_test_split(
            X, y, idx, test_size=0.4, random_state=random_state
        )
        X_val, X_test, y_val, y_test, _, i_test = train_test_split(
            X_temp, y_temp, i_temp, test_size=0.5, random_state=random_state
        )

        model = TransformedTargetRegressor(
            regressor=HistGradientBoostingRegressor(max_iter=200, random_state=random_state),
            func=np.log1p,
            inverse_func=np.expm1,
        )
        model.fit(X_train, y_train)
        logger.info(
            "Trained %s [%s]  train=%d  val=%d  test=%d",
            phase,
            target_mode,
            len(y_train),
            len(y_val),
            len(y_test),
        )

        test_nct = nct[i_test]
        cmap = _category_map(df)
        categories = np.array([cmap.get(str(n), "") for n in test_nct], dtype=object)

        results[phase] = {
            "model": model,
            "X_test": X_test,
            "y_test": y_test,
            "nct_id": test_nct,
            "category": categories,
            "target_mode": target_mode,
        }

    return results


def deviation_table_from_phase_models(
    phase_results: dict[str, dict[str, Any]],
    *,
    threshold_pct: float,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for phase, data in phase_results.items():
        predicted_days = data["model"].predict(data["X_test"])
        actual_days = data["y_test"]
        pct_dev = calculate_pct_deviation(actual_days, predicted_days)
        abs_error = np.abs(actual_days - predicted_days)
        late = make_late_flag(pct_dev, threshold_pct)

        frame = pd.DataFrame(
            {
                "nct_id": data["nct_id"],
                "phase": phase,
                "split": "test",
                "analysis_target": data.get("target_mode", ""),
                "category": data["category"],
                "actual_days": actual_days,
                "predicted_days": predicted_days,
                "pct_deviation": pct_dev,
                "abs_error_days": abs_error,
                "late_flag": late,
            }
        )
        frames.append(frame)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def deviation_table_from_combined_csv(
    path: Path,
    *,
    threshold_pct: float,
    splits: tuple[str, ...] | None,
) -> pd.DataFrame:
    raw = pd.read_csv(path, low_memory=False)
    if "split" in raw.columns and splits is not None:
        raw = raw[raw["split"].astype(str).isin(splits)].copy()

    cmap = _category_map(raw) if "nct_id" in raw.columns else {}
    frames: list[pd.DataFrame] = []

    for kind, act_c, pred_c in COMBINED_SPECS:
        if act_c not in raw.columns or pred_c not in raw.columns:
            logger.warning("Combined CSV missing columns %s / %s — skip %s", act_c, pred_c, kind)
            continue
        m = raw[act_c].notna() & raw[pred_c].notna()
        sub = raw.loc[m].copy()
        if sub.empty:
            continue
        a = sub[act_c].to_numpy(dtype=np.float64)
        p = sub[pred_c].to_numpy(dtype=np.float64)
        pct_dev = calculate_pct_deviation(a, p)
        late = make_late_flag(pct_dev, threshold_pct)
        nct = sub["nct_id"].astype(str).to_numpy() if "nct_id" in sub.columns else np.full(len(sub), "")
        phase = sub["phase"].astype(str).to_numpy() if "phase" in sub.columns else np.full(len(sub), "")
        spl = (
            sub["split"].astype(str).to_numpy()
            if "split" in sub.columns
            else np.full(len(sub), "unknown")
        )
        cat = np.array([cmap.get(str(n), "") for n in nct], dtype=object)
        frames.append(
            pd.DataFrame(
                {
                    "nct_id": nct,
                    "phase": phase,
                    "split": spl,
                    "analysis_target": kind,
                    "category": cat,
                    "actual_days": a,
                    "predicted_days": p,
                    "pct_deviation": pct_dev,
                    "abs_error_days": np.abs(a - p),
                    "late_flag": late,
                }
            )
        )

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def default_output_paths(target: str) -> tuple[Path, Path]:
    if target == "primary_completion":
        return RESULTS_DIR / "predictions_with_deviation.csv", RESULTS_DIR / "deviation_summary.txt"
    return (
        RESULTS_DIR / f"predictions_with_deviation_{target}.csv",
        RESULTS_DIR / f"deviation_summary_{target}.txt",
    )


def run_analysis(
    *,
    target: TargetMode,
    threshold_pct: float,
    random_state: int,
    combined_csv: Path | None,
    output_csv: Path | None,
    output_summary: Path | None,
    splits: tuple[str, ...] | None,
) -> None:
    out_csv = output_csv or default_output_paths(target)[0]
    out_summary = output_summary or default_output_paths(target)[1]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_summary.parent.mkdir(parents=True, exist_ok=True)

    if target == "combined":
        cpath = combined_csv or (RESULTS_DIR / "combined_duration_predictions.csv")
        if not cpath.exists():
            raise FileNotFoundError(
                f"Combined predictions not found: {cpath}. Run 4_regression/combined_duration_forecast.py first."
            )
        deviation_df = deviation_table_from_combined_csv(
            cpath, threshold_pct=threshold_pct, splits=splits
        )
        title = "PREDICTION DEVIATION ANALYSIS (combined staged forecasts)"
        header_extra = (
            f"Source predictions: {cpath}",
            f"Splits filter: {splits or 'all'}",
        )
    else:
        logger.info("Loading cohort via cohort_io.load_and_join ...")
        df = load_and_join(
            eligibility_columns=KEPT_ELIGIBILITY,
            site_footprint_columns=KEPT_SITE_FOOTPRINT,
            design_columns=KEPT_DESIGN,
            arm_intervention_columns=KEPT_ARM_INTERVENTION,
            design_outcomes_columns=KEPT_DESIGN_OUTCOMES,
        )
        # load_and_join already COMPLETED-only
        logger.info("Training dedicated phase models for target=%s ...", target)
        phase_results = train_dedicated_phase_models(df, target_mode=target, random_state=random_state)
        deviation_df = deviation_table_from_phase_models(phase_results, threshold_pct=threshold_pct)
        _, formula = describe_target_kind(REGRESSION_TARGET_CONFIG[target][1])
        pol = REGRESSION_TARGET_CONFIG[target][0]
        title = f"PREDICTION DEVIATION ANALYSIS ({target})"
        header_extra = (
            f"Feature policy: {pol}",
            f"Duration definition: {formula}",
            "Model: dedicated PHASE1/2/3 only (test split); same split as historical baseline script",
        )

    if deviation_df.empty:
        logger.warning("No deviation rows produced.")
        deviation_df.to_csv(out_csv, index=False)
        out_summary.write_text("No data.\n")
        return

    if target != "combined" and splits != ("test",):
        logger.warning("--splits is only applied for --target combined; regression outputs are test-only.")

    deviation_df.to_csv(out_csv, index=False)
    logger.info("Saved %s", out_csv)

    report = format_deviation_summary_report(
        deviation_df,
        phase_order=list(PHASE_REPORT_ORDER),
        late_threshold_pct=threshold_pct,
        title=title,
        header_extra=header_extra,
        group_col="analysis_target" if target == "combined" else None,
    )
    out_summary.write_text(report)
    logger.info("Saved %s", out_summary)
    print(report)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-target prediction deviation analysis")
    p.add_argument(
        "--target",
        choices=("primary_completion", "post_primary_completion", "total_completion", "combined"),
        default="primary_completion",
        help="Which duration target or combined CSV (default: primary_completion).",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=20.0,
        help="Percent deviation above prediction to flag as late (default: 20)",
    )
    p.add_argument("--random-state", type=int, default=42, help="Split + HGBR seed for regression targets")
    p.add_argument(
        "--combined-csv",
        type=Path,
        default=None,
        help="For --target combined, path to combined_duration_predictions.csv",
    )
    p.add_argument("--output-csv", type=Path, default=None, help="Override output predictions CSV path")
    p.add_argument("--output-summary", type=Path, default=None, help="Override output summary txt path")
    p.add_argument(
        "--splits",
        type=str,
        default="test",
        help="For combined target: comma-separated splits to include (default: test). Use 'all' for every split.",
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _parse_args()
    sp = args.splits.strip().lower()
    splits: tuple[str, ...] | None
    if sp == "all":
        splits = None
    else:
        splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())

    run_analysis(
        target=args.target,
        threshold_pct=args.threshold,
        random_state=args.random_state,
        combined_csv=args.combined_csv,
        output_csv=args.output_csv,
        output_summary=args.output_summary,
        splits=splits,
    )


if __name__ == "__main__":
    main()
