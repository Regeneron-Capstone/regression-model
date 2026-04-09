#!/usr/bin/env python3
"""
Late-risk classification (separate from regression): predict high tail duration vs peers.

- Features: strict_planning only (via ``prepare_features`` / ``assemble_feature_matrix``).
- Model: HistGradientBoostingClassifier (reproducible ``random_state``).
- Label: per-phase quantile of *training-split* total completion days (start → study completion);
  positive if actual duration exceeds that phase's training quantile (fallback: global train quantile).

Outputs:
  results/late_risk_classification_report.txt
  results/late_risk_predictions.csv

Usage (repo root):
  python 4_regression/late_risk_classifier.py
  python 4_regression/late_risk_classifier.py --late-quantile 0.80 --random-state 42
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from cohort_columns import (  # noqa: E402
    KEPT_ARM_INTERVENTION,
    KEPT_DESIGN,
    KEPT_DESIGN_OUTCOMES,
    KEPT_ELIGIBILITY,
    KEPT_ELIGIBILITY_CRITERIA_TEXT,
    KEPT_SITE_FOOTPRINT,
)
from cohort_io import load_and_join  # noqa: E402
from train_regression import RESULTS_DIR, prepare_features  # noqa: E402

logger = logging.getLogger(__name__)


def _prep_kw_strict() -> dict:
    return dict(
        eligibility_columns=KEPT_ELIGIBILITY,
        eligibility_criteria_text_columns=KEPT_ELIGIBILITY_CRITERIA_TEXT,
        site_footprint_columns=KEPT_SITE_FOOTPRINT,
        design_columns=KEPT_DESIGN,
        arm_intervention_columns=KEPT_ARM_INTERVENTION,
        design_outcomes_columns=KEPT_DESIGN_OUTCOMES,
        encode_phase=False,
        policy="strict_planning",
        target_kind="total_completion",
    )


def _phase_quantile_thresholds(
    y_train: np.ndarray,
    phase_train: np.ndarray,
    quantile: float,
    *,
    min_phase_rows: int = 30,
) -> tuple[dict[str, float], float]:
    """Quantile of total duration per phase on train; phases with few rows use global train quantile."""
    df_t = pd.DataFrame({"y": y_train.astype(np.float64), "phase": phase_train.astype(str)})
    global_thr = float(df_t["y"].quantile(quantile))
    thresholds: dict[str, float] = {}
    for ph, g in df_t.groupby("phase"):
        if len(g) >= min_phase_rows:
            thresholds[ph] = float(g["y"].quantile(quantile))
        else:
            thresholds[ph] = global_thr
    return thresholds, global_thr


def _binary_late_labels(
    y: np.ndarray,
    phases: np.ndarray,
    thresholds: dict[str, float],
    global_thr: float,
) -> np.ndarray:
    thr = pd.Series(phases.astype(str)).map(thresholds).fillna(global_thr).to_numpy(dtype=np.float64)
    return (y > thr).astype(np.int32)


def _safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    y_true = np.asarray(y_true, dtype=int)
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def _safe_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    y_true = np.asarray(y_true, dtype=int)
    if len(np.unique(y_true)) < 2:
        return None
    return float(average_precision_score(y_true, y_score))


def _metrics_block(name: str, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> list[str]:
    lines = [f"Split: {name}", f"  n = {len(y_true):,}"]
    pos_rate = float(np.mean(y_true)) if len(y_true) else 0.0
    lines.append(f"  positive rate (late) = {pos_rate:.4f}")

    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    lines.append(f"  precision = {prec:.4f}")
    lines.append(f"  recall    = {rec:.4f}")
    lines.append(f"  F1        = {f1:.4f}")

    roc = _safe_roc_auc(y_true, y_proba)
    pr = _safe_pr_auc(y_true, y_proba)
    lines.append(f"  ROC-AUC   = {roc:.4f}" if roc is not None else "  ROC-AUC   = n/a (single class)")
    lines.append(f"  PR-AUC    = {pr:.4f}" if pr is not None else "  PR-AUC    = n/a (single class)")
    lines.append("")
    return lines


def run(
    *,
    late_quantile: float,
    random_state: int,
    report_path: Path,
    predictions_path: Path,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_and_join(
        eligibility_columns=KEPT_ELIGIBILITY,
        site_footprint_columns=KEPT_SITE_FOOTPRINT,
        design_columns=KEPT_DESIGN,
        arm_intervention_columns=KEPT_ARM_INTERVENTION,
        design_outcomes_columns=KEPT_DESIGN_OUTCOMES,
    )
    # load_and_join already restricts to COMPLETED trials
    cohort = df.copy()
    cohort["phase"] = cohort["phase"].astype(str)

    logger.info("Cohort rows (COMPLETED): %s", f"{len(cohort):,}")

    prep = _prep_kw_strict()
    X, y_cont, phases, art = prepare_features(cohort, **prep)
    nct_ids = art.get("nct_ids")
    if nct_ids is None:
        raise RuntimeError("Expected nct_ids in feature artifacts; rebuild with current features.py")

    logger.info(
        "After total_completion + strict_planning matrix: %s rows",
        f"{len(y_cont):,}",
    )

    idx = np.arange(len(y_cont))
    i_train, i_temp = train_test_split(
        idx, test_size=0.4, random_state=random_state, shuffle=True
    )
    i_val, i_test = train_test_split(
        i_temp, test_size=0.5, random_state=random_state, shuffle=True
    )

    thresholds, global_thr = _phase_quantile_thresholds(
        y_cont[i_train], phases[i_train], late_quantile
    )

    y_train = _binary_late_labels(y_cont[i_train], phases[i_train], thresholds, global_thr)
    y_val = _binary_late_labels(y_cont[i_val], phases[i_val], thresholds, global_thr)
    y_test = _binary_late_labels(y_cont[i_test], phases[i_test], thresholds, global_thr)

    clf = HistGradientBoostingClassifier(
        max_iter=200,
        random_state=random_state,
        class_weight="balanced",
    )
    clf.fit(X[i_train], y_train)

    def _proba_pos(Xs: np.ndarray) -> np.ndarray:
        p = clf.predict_proba(Xs)
        return p[:, 1] if p.shape[1] > 1 else p[:, 0]

    proba_train = _proba_pos(X[i_train])
    proba_val = _proba_pos(X[i_val])
    proba_test = _proba_pos(X[i_test])
    pred_train = (proba_train >= 0.5).astype(int)
    pred_val = (proba_val >= 0.5).astype(int)
    pred_test = (proba_test >= 0.5).astype(int)

    lines: list[str] = []
    lines.append("LATE-RISK CLASSIFICATION (strict planning features only)")
    lines.append("=" * 60)
    lines.append("Model: HistGradientBoostingClassifier (max_iter=200, class_weight=balanced)")
    lines.append(f"random_state: {random_state}")
    lines.append(f"Feature policy: strict_planning (no regression trainer coupling)")
    lines.append(f"Regression target used only to define rows/continuity: total_completion (days)")
    lines.append("")
    lines.append("Label definition")
    lines.append(f"  late_risk = 1  iff  actual_total_completion_days > Q{late_quantile:.2f} within phase")
    lines.append("  Quantiles are fit on TRAIN split only; same thresholds applied to val/test.")
    lines.append(f"  Phases with < 30 train rows use global train Q{late_quantile:.2f}.")
    lines.append(f"  Global train threshold (fallback) = {global_thr:.1f} days")
    lines.append("")
    lines.append(f"Train / val / test sizes: {len(i_train):,} / {len(i_val):,} / {len(i_test):,}")
    lines.append(
        f"Train positive rate: {y_train.mean():.4f}  "
        f"val: {y_val.mean():.4f}  test: {y_test.mean():.4f}"
    )
    lines.append("")
    lines.extend(_metrics_block("train (in-sample)", y_train, pred_train, proba_train))
    lines.extend(_metrics_block("val", y_val, pred_val, proba_val))
    lines.extend(_metrics_block("test", y_test, pred_test, proba_test))
    lines.append("End of report")
    lines.append("=" * 60)

    report_text = "\n".join(lines)
    report_path.write_text(report_text, encoding="utf-8")
    logger.info("Wrote report %s", report_path)
    print(report_text)

    split_names = np.empty(len(y_cont), dtype=object)
    split_names[i_train] = "train"
    split_names[i_val] = "val"
    split_names[i_test] = "test"

    proba_all = _proba_pos(X)
    pred_all = (proba_all >= 0.5).astype(int)
    thr_row = pd.Series(phases.astype(str)).map(thresholds).fillna(global_thr).to_numpy(dtype=np.float64)
    y_bin_all = _binary_late_labels(y_cont, phases, thresholds, global_thr)

    out = pd.DataFrame(
        {
            "nct_id": nct_ids,
            "phase": phases.astype(str),
            "split": split_names,
            "actual_total_completion_days": y_cont.astype(np.float64),
            "late_threshold_days": thr_row,
            "late_risk_true": y_bin_all,
            "late_risk_pred_proba": proba_all,
            "late_risk_pred": pred_all,
        }
    )
    out.to_csv(predictions_path, index=False)
    logger.info("Wrote predictions %s (%s rows)", predictions_path, f"{len(out):,}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Late-risk classifier (strict planning features).")
    p.add_argument(
        "--late-quantile",
        type=float,
        default=0.75,
        help="Per-phase quantile on train for late label (default: 0.75).",
    )
    p.add_argument("--random-state", type=int, default=42, help="Split + HGBC seed (default: 42).")
    p.add_argument(
        "--report",
        type=Path,
        default=RESULTS_DIR / "late_risk_classification_report.txt",
        help="Evaluation report path.",
    )
    p.add_argument(
        "--predictions",
        type=Path,
        default=RESULTS_DIR / "late_risk_predictions.csv",
        help="Per-trial prediction CSV path.",
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _parse_args()
    if not (0.0 < args.late_quantile < 1.0):
        raise SystemExit("--late-quantile must be in (0, 1)")
    run(
        late_quantile=args.late_quantile,
        random_state=args.random_state,
        report_path=args.report.expanduser().resolve(),
        predictions_path=args.predictions.expanduser().resolve(),
    )


if __name__ == "__main__":
    main()
