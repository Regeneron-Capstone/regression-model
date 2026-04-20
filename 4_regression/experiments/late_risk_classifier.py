#!/usr/bin/env python3
"""
Late-risk classification (separate from regression): predict high tail duration vs peers.

- Features: strict_planning only (via ``prepare_features`` / ``assemble_feature_matrix``).
- Model: HistGradientBoostingClassifier (reproducible ``random_state``).
- Label: per-(phase, disease category) quantile of *training-split* total completion days
  (start → study completion); positive if actual duration exceeds that cell's threshold.
  Disease category is the CCSR domain (``ccsr_domain`` — 21 body-system categories);
  unmapped trials use the ``Other_Unclassified`` bucket (consistent with the regression
  cohort). Hierarchical fallback for sparse cells:
    (phase, domain) → phase → global, each gated on ``--min-group-rows``.
  Use ``--disease-axis none`` to reproduce the previous phase-only label for A/B comparison.

Outputs:
  6_results/late_risk_classification_report.txt
  6_results/late_risk_predictions.csv

Usage (repo root):
  python 4_regression/experiments/late_risk_classifier.py
  python 4_regression/experiments/late_risk_classifier.py --late-quantile 0.80 --random-state 42
  python 4_regression/experiments/late_risk_classifier.py --disease-axis none   # phase-only A/B
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

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
    KEPT_ELIGIBILITY_CRITERIA_TEXT,
    KEPT_SITE_FOOTPRINT,
    default_feature_prep_kw,
)
from step00_cohort_io import load_and_join  # noqa: E402
from step03_train_regression import RESULTS_DIR, prepare_features  # noqa: E402

logger = logging.getLogger(__name__)

DiseaseAxis = Literal["ccsr_domain", "none"]

UNCLASSIFIED_CATEGORY = "Other_Unclassified"
PHASE_ONLY_DOMAIN = "_ALL_"  # sentinel when --disease-axis none


@dataclass(frozen=True)
class ThresholdMap:
    """Hierarchical late-risk thresholds fit on the training split.

    - ``group`` keyed on ``(phase, domain)`` when that cell has enough train rows.
    - ``phase`` keyed on phase when the (phase, domain) cell is too sparse.
    - ``global_thr`` is the final fallback (always available).
    - ``group_counts`` carries the train-row count for each (phase, domain) cell.
    - ``source`` records, per (phase, domain) cell, which rule produced its threshold:
      ``"group" | "phase" | "global"``.
    """

    group: dict[tuple[str, str], float]
    phase: dict[str, float]
    global_thr: float
    group_counts: dict[tuple[str, str], int]
    phase_counts: dict[str, int]
    source: dict[tuple[str, str], str]
    min_group_rows: int
    quantile: float
    disease_axis: DiseaseAxis

    def lookup(self, phase: str, domain: str) -> tuple[float, str]:
        """Return ``(threshold_days, source)`` for an arbitrary (phase, domain) row.

        Unseen (phase, domain) cells fall back the same way as sparse train cells:
        use phase Q75 when available, otherwise the global Q75.
        """
        phase_s = str(phase)
        domain_s = str(domain) if domain is not None else UNCLASSIFIED_CATEGORY
        key = (phase_s, domain_s)
        if key in self.group:
            return self.group[key], self.source.get(key, "group")
        if phase_s in self.phase:
            return self.phase[phase_s], "phase"
        return self.global_thr, "global"


def _prep_kw_strict() -> dict:
    return default_feature_prep_kw(policy="strict_planning", target_kind="total_completion")


def _fit_threshold_map(
    y_train: np.ndarray,
    phase_train: np.ndarray,
    domain_train: np.ndarray,
    *,
    quantile: float,
    min_group_rows: int,
    disease_axis: DiseaseAxis,
) -> ThresholdMap:
    """Fit hierarchical (phase, domain) → phase → global Q-quantile thresholds on TRAIN only."""
    df_t = pd.DataFrame(
        {
            "y": y_train.astype(np.float64),
            "phase": phase_train.astype(str),
            "domain": domain_train.astype(str),
        }
    )
    global_thr = float(df_t["y"].quantile(quantile))

    phase_thr: dict[str, float] = {}
    phase_counts: dict[str, int] = {}
    for ph, g in df_t.groupby("phase"):
        phase_counts[ph] = int(len(g))
        if len(g) >= min_group_rows:
            phase_thr[ph] = float(g["y"].quantile(quantile))

    group_thr: dict[tuple[str, str], float] = {}
    group_counts: dict[tuple[str, str], int] = {}
    source: dict[tuple[str, str], str] = {}

    if disease_axis == "none":
        for ph, n in phase_counts.items():
            key = (ph, PHASE_ONLY_DOMAIN)
            group_counts[key] = n
            if ph in phase_thr:
                group_thr[key] = phase_thr[ph]
                source[key] = "phase"
            else:
                group_thr[key] = global_thr
                source[key] = "global"
    else:
        for (ph, dom), g in df_t.groupby(["phase", "domain"]):
            key = (ph, dom)
            n = int(len(g))
            group_counts[key] = n
            if n >= min_group_rows:
                group_thr[key] = float(g["y"].quantile(quantile))
                source[key] = "group"
            elif ph in phase_thr:
                group_thr[key] = phase_thr[ph]
                source[key] = "phase"
            else:
                group_thr[key] = global_thr
                source[key] = "global"

    return ThresholdMap(
        group=group_thr,
        phase=phase_thr,
        global_thr=global_thr,
        group_counts=group_counts,
        phase_counts=phase_counts,
        source=source,
        min_group_rows=min_group_rows,
        quantile=quantile,
        disease_axis=disease_axis,
    )


def _apply_threshold_map(
    y: np.ndarray,
    phases: np.ndarray,
    domains: np.ndarray,
    tmap: ThresholdMap,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized lookup: returns (late_labels, per_row_thresholds, per_row_sources)."""
    n = len(y)
    thr_row = np.empty(n, dtype=np.float64)
    src_row = np.empty(n, dtype=object)
    phase_s = phases.astype(str)
    domain_s = domains.astype(str)
    for i in range(n):
        thr, src = tmap.lookup(phase_s[i], domain_s[i])
        thr_row[i] = thr
        src_row[i] = src
    labels = (y.astype(np.float64) > thr_row).astype(np.int32)
    return labels, thr_row, src_row


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


def _threshold_table_block(
    tmap: ThresholdMap,
    y_train: np.ndarray,
    phase_train: np.ndarray,
    domain_train: np.ndarray,
    train_labels: np.ndarray,
) -> list[str]:
    """Render the per-group threshold table ordered by (phase, domain)."""
    lines: list[str] = []
    lines.append("Per-group threshold table (fit on TRAIN split only)")
    lines.append("-" * 80)
    if tmap.disease_axis == "none":
        header = f"{'phase':<18}{'n_train':>10}  {'Q':>5}%  {'threshold_days':>15}  {'source':<8}  {'pos_rate':>9}"
    else:
        header = (
            f"{'phase':<18}{'domain':<10}{'n_train':>10}  {'Q':>5}%  "
            f"{'threshold_days':>15}  {'source':<8}  {'pos_rate':>9}"
        )
    lines.append(header)
    lines.append("-" * 80)

    df = pd.DataFrame(
        {
            "phase": phase_train.astype(str),
            "domain": domain_train.astype(str),
            "y_train_late": train_labels.astype(int),
        }
    )
    if tmap.disease_axis == "none":
        df["domain"] = PHASE_ONLY_DOMAIN
    pos_rates = df.groupby(["phase", "domain"])["y_train_late"].mean().to_dict()

    pct = int(round(tmap.quantile * 100))
    for key in sorted(tmap.group.keys()):
        phase, domain = key
        n = tmap.group_counts.get(key, 0)
        thr = tmap.group[key]
        src = tmap.source.get(key, "group")
        pr = pos_rates.get(key, float("nan"))
        pr_str = f"{pr:.3f}" if not np.isnan(pr) else "   -   "
        if tmap.disease_axis == "none":
            lines.append(
                f"{phase:<18}{n:>10,}  {pct:>4}%  {thr:>15,.1f}  {src:<8}  {pr_str:>9}"
            )
        else:
            lines.append(
                f"{phase:<18}{domain:<10}{n:>10,}  {pct:>4}%  {thr:>15,.1f}  {src:<8}  {pr_str:>9}"
            )
    lines.append("-" * 80)
    lines.append(
        f"Fallback hierarchy: (phase, domain) Q{tmap.quantile:.2f}  "
        f"→ phase Q{tmap.quantile:.2f}  →  global Q{tmap.quantile:.2f} "
        f"(min_group_rows={tmap.min_group_rows})."
    )
    lines.append(f"Global train threshold (final fallback) = {tmap.global_thr:,.1f} days")
    if tmap.phase:
        lines.append("Phase-level Q thresholds (days): " + ", ".join(
            f"{p}={t:,.1f}" for p, t in sorted(tmap.phase.items())
        ))
    lines.append("")
    return lines


def _align_domains(
    cohort: pd.DataFrame,
    nct_ids: np.ndarray,
    disease_axis: DiseaseAxis,
) -> np.ndarray:
    """Return per-row disease category aligned to the post-feature-prep nct_ids."""
    if disease_axis == "none":
        return np.full(len(nct_ids), PHASE_ONLY_DOMAIN, dtype=object)
    if "category" in cohort.columns:
        src_col = "category"
    elif "ccsr_domain" in cohort.columns:
        src_col = "ccsr_domain"
    else:
        logger.warning(
            "No 'category' or 'ccsr_domain' column on cohort; treating all rows as %s",
            UNCLASSIFIED_CATEGORY,
        )
        return np.full(len(nct_ids), UNCLASSIFIED_CATEGORY, dtype=object)

    mapper = (
        cohort[["nct_id", src_col]]
        .drop_duplicates("nct_id")
        .set_index("nct_id")[src_col]
    )
    out = (
        pd.Series(nct_ids).astype(str).map(mapper).fillna(UNCLASSIFIED_CATEGORY).astype(str)
    )
    return out.to_numpy(dtype=object)


def run(
    *,
    late_quantile: float,
    random_state: int,
    report_path: Path,
    predictions_path: Path,
    disease_axis: DiseaseAxis = "ccsr_domain",
    min_group_rows: int = 30,
    decision_threshold: float = 0.6,
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
    cohort = df.copy()
    cohort["phase"] = cohort["phase"].astype(str)

    logger.info("Cohort rows (COMPLETED): %s", f"{len(cohort):,}")

    prep = _prep_kw_strict()
    X, y_cont, phases, art = prepare_features(cohort, **prep)
    nct_ids = art.get("nct_ids")
    if nct_ids is None:
        raise RuntimeError("Expected nct_ids in feature artifacts; rebuild with current features.py")

    domains = _align_domains(cohort, nct_ids, disease_axis)

    logger.info(
        "After total_completion + strict_planning matrix: %s rows",
        f"{len(y_cont):,}",
    )
    if disease_axis != "none":
        n_unclassified = int(np.sum(domains == UNCLASSIFIED_CATEGORY))
        logger.info(
            "Disease-axis=%s; %s unique domains; %s rows fall into %s.",
            disease_axis,
            len(pd.unique(domains)),
            f"{n_unclassified:,}",
            UNCLASSIFIED_CATEGORY,
        )

    idx = np.arange(len(y_cont))
    i_train, i_temp = train_test_split(
        idx, test_size=0.4, random_state=random_state, shuffle=True
    )
    i_val, i_test = train_test_split(
        i_temp, test_size=0.5, random_state=random_state, shuffle=True
    )

    tmap = _fit_threshold_map(
        y_cont[i_train],
        phases[i_train],
        domains[i_train],
        quantile=late_quantile,
        min_group_rows=min_group_rows,
        disease_axis=disease_axis,
    )

    y_train, thr_train, src_train = _apply_threshold_map(
        y_cont[i_train], phases[i_train], domains[i_train], tmap
    )
    y_val, thr_val, src_val = _apply_threshold_map(
        y_cont[i_val], phases[i_val], domains[i_val], tmap
    )
    y_test, thr_test, src_test = _apply_threshold_map(
        y_cont[i_test], phases[i_test], domains[i_test], tmap
    )

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
    pred_train = (proba_train >= decision_threshold).astype(int)
    pred_val = (proba_val >= decision_threshold).astype(int)
    pred_test = (proba_test >= decision_threshold).astype(int)

    axis_desc = (
        "phase only (legacy)"
        if disease_axis == "none"
        else f"(phase, {disease_axis})"
    )

    lines: list[str] = []
    lines.append("LATE-RISK CLASSIFICATION (strict planning features only)")
    lines.append("=" * 72)
    lines.append("Model: HistGradientBoostingClassifier (max_iter=200, class_weight=balanced)")
    lines.append(f"random_state: {random_state}")
    lines.append(f"Positive-class decision threshold (predicted probability): {decision_threshold:g}")
    lines.append("Feature policy: strict_planning (no regression trainer coupling)")
    lines.append("Regression target used only to define rows/continuity: total_completion (days)")
    lines.append("")
    lines.append("Label definition")
    lines.append(
        f"  late_risk = 1  iff  actual_total_completion_days > Q{late_quantile:.2f} "
        f"within {axis_desc}"
    )
    lines.append(f"  Disease axis: {disease_axis}  |  min_group_rows: {min_group_rows}")
    lines.append(
        "  Thresholds fit on TRAIN only; hierarchical fallback to phase, then global, "
        "for sparse cells."
    )
    lines.append(
        "  Unseen (phase, domain) cells at inference: fall back to phase Q, else global Q."
    )
    lines.append("")

    lines.extend(
        _threshold_table_block(tmap, y_cont[i_train], phases[i_train], domains[i_train], y_train)
    )

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
    lines.append("=" * 72)

    report_text = "\n".join(lines)
    report_path.write_text(report_text, encoding="utf-8")
    logger.info("Wrote report %s", report_path)
    print(report_text)

    split_names = np.empty(len(y_cont), dtype=object)
    split_names[i_train] = "train"
    split_names[i_val] = "val"
    split_names[i_test] = "test"

    proba_all = _proba_pos(X)
    pred_all = (proba_all >= decision_threshold).astype(int)
    y_all, thr_all, src_all = _apply_threshold_map(y_cont, phases, domains, tmap)

    out = pd.DataFrame(
        {
            "nct_id": nct_ids,
            "phase": phases.astype(str),
            "disease_category": np.asarray(domains, dtype=object),
            "split": split_names,
            "actual_total_completion_days": y_cont.astype(np.float64),
            "late_threshold_days": thr_all,
            "late_threshold_source": np.asarray(src_all, dtype=object),
            "late_risk_true": y_all,
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
        help="Within-group quantile on train for late label (default: 0.75).",
    )
    p.add_argument(
        "--disease-axis",
        choices=("ccsr_domain", "none"),
        default="ccsr_domain",
        help=(
            "Disease-category axis for stratified thresholds. "
            "'ccsr_domain' (default) uses 21 CCSR body-system categories; "
            "'none' reproduces the previous phase-only thresholds."
        ),
    )
    p.add_argument(
        "--min-group-rows",
        type=int,
        default=30,
        help=(
            "Minimum training rows required in a (phase, domain) cell before its own "
            "quantile is used; sparser cells fall back to phase, then global (default: 30)."
        ),
    )
    p.add_argument("--random-state", type=int, default=42, help="Split + HGBC seed (default: 42).")
    p.add_argument(
        "--decision-threshold",
        type=float,
        default=0.6,
        help="Probability threshold for late_risk_pred=1 (default: 0.6).",
    )
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
    if args.min_group_rows < 1:
        raise SystemExit("--min-group-rows must be >= 1")
    if not (0.0 < args.decision_threshold < 1.0):
        raise SystemExit("--decision-threshold must be in (0, 1)")
    run(
        late_quantile=args.late_quantile,
        random_state=args.random_state,
        report_path=args.report.expanduser().resolve(),
        predictions_path=args.predictions.expanduser().resolve(),
        disease_axis=args.disease_axis,
        min_group_rows=args.min_group_rows,
        decision_threshold=args.decision_threshold,
    )


if __name__ == "__main__":
    main()
