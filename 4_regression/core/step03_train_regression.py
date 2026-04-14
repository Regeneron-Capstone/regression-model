"""Step 03: Train phase-specific regression models and generate evaluation reports."""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split

_SCRIPT_DIR = Path(__file__).resolve().parent
_REGRESSION_DIR = _SCRIPT_DIR.parent
if str(_REGRESSION_DIR) not in sys.path:
    sys.path.insert(0, str(_REGRESSION_DIR))
from step04_evaluation import (
    evaluate_sklearn_split,
    evaluate_subset,
    joint_subset_report_line,
    metrics_report_line,
    mixed_cohort_test_line,
)
from cohort_columns import (
    EARLY_JOINT_PHASES,
    KEPT_ARM_INTERVENTION,
    KEPT_DESIGN,
    KEPT_DESIGN_OUTCOMES,
    KEPT_ELIGIBILITY,
    KEPT_ELIGIBILITY_CRITERIA_TEXT,
    KEPT_SITE_FOOTPRINT,
    LATE_JOINT_PHASES,
    PHASE_REPORT_ORDER,
    PHASE_SINGLE_MODELS,
    PHASES_WITH_DEDICATED_MODELS,
)
from step00_cohort_io import load_and_join
from step01_features import assemble_feature_matrix

from step02_targets import DEFAULT_TARGET_KIND, TARGET_DURATION_COLUMN, describe_target_kind

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "6_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Features: target_duration (97% null) and number_of_groups (100% null) excluded
# category (132 unique) outperforms therapeutic_area (16 unique): R² 0.317 vs 0.285
# Phase: one-hot (R² 0.317) > phase flags (0.315) > no phase (0.280)
# downcase_mesh_term: ablation R² 0.319 vs 0.317 baseline — small gain, included
# intervention_type: ablation R² 0.320 vs 0.319 baseline — included
# eligibility: gender, minimum_age, maximum_age, adult, child, older_adult (ablation-tested)
FEATURE_COLUMNS = [
    "phase",
    "enrollment",
    "n_sponsors",
    "number_of_arms",
    "start_year",
    "category",
    "downcase_mesh_term",
    "intervention_type",
]
ELIGIBILITY_COLUMNS = ["gender", "minimum_age", "maximum_age", "adult", "child", "older_adult"]
SITE_FOOTPRINT_FEATURES = [
    "number_of_facilities",
    "number_of_countries",
    "us_only",
    "number_of_us_states",
    "has_single_facility",
    "facility_density",
]
DESIGN_FEATURES = [
    "randomized",
    "intervention_model",
    "masking_depth_score",
    "primary_purpose",
    "design_complexity_composite",
]
ARM_INTERVENTION_FEATURES = [
    "number_of_interventions",
    "intervention_type_diversity",
    "mono_therapy",
    "has_placebo",
    "has_active_comparator",
    "n_mesh_intervention_terms",
]
# Primary-completion span column on clean studies (preprocess); also the default regression target.
TARGET_COLUMN = TARGET_DURATION_COLUMN

TARGET_KIND_CHOICES: tuple[str, ...] = (
    "primary_completion",
    "post_primary_completion",
    "total_completion",
)
FEATURE_POLICY_CHOICES: tuple[str, ...] = ("baseline", "strict_planning")


def resolve_report_path(
    target_kind: str,
    feature_policy: str,
    report_arg: Path | None,
) -> Path:
    """Default report path preserves legacy ``regression_report.txt`` for baseline primary target."""
    if report_arg is not None:
        return report_arg.expanduser().resolve()
    if target_kind == DEFAULT_TARGET_KIND and feature_policy == "baseline":
        return RESULTS_DIR / "regression_report.txt"
    return RESULTS_DIR / f"regression_report_{target_kind}_{feature_policy}.txt"


def prepare_features(
    df: pd.DataFrame,
    eligibility_columns: list[str] | None = None,
    eligibility_criteria_text_columns: list[str] | None = None,
    site_footprint_columns: list[str] | None = None,
    design_columns: list[str] | None = None,
    arm_intervention_columns: list[str] | None = None,
    design_outcomes_columns: list[str] | None = None,
    *,
    encode_phase: bool = False,
    policy: str = "baseline",
    target_kind: str = DEFAULT_TARGET_KIND,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Build X (float matrix with NaN allowed for HGBR), y, phase labels, and encoders.
    Delegates to ``features.assemble_feature_matrix``; ``policy="baseline"`` preserves
    historical training behavior. ``target_kind`` selects y via ``targets.resolve_target_series``.
    """
    return assemble_feature_matrix(
        df,
        eligibility_columns=eligibility_columns,
        eligibility_criteria_text_columns=eligibility_criteria_text_columns,
        site_footprint_columns=site_footprint_columns,
        design_columns=design_columns,
        arm_intervention_columns=arm_intervention_columns,
        design_outcomes_columns=design_outcomes_columns,
        encode_phase=encode_phase,
        policy=policy,  # type: ignore[arg-type]
        target_kind=target_kind,
    )


def _joint_test_metrics_by_phase(
    model: TransformedTargetRegressor,
    X_test: np.ndarray,
    y_test: np.ndarray,
    ph_test: np.ndarray,
    phase_labels: tuple[str, ...],
) -> dict[str, dict]:
    """Test RMSE/MAE/R² per phase label on the joint model's test fold (need n>=2 for R²)."""
    out: dict[str, dict] = {}
    for ph in phase_labels:
        mask = ph_test == ph
        y_s = y_test[mask]
        X_s = X_test[mask]
        if len(y_s) < 2:
            continue
        ev = evaluate_subset(model, X_s, y_s)
        out[ph] = {
            "n": ev["n"],
            "r2": ev["r2"],
            "rmse": ev["rmse"],
            "mae": ev["mae"],
        }
    return out


def _new_regressor() -> TransformedTargetRegressor:
    return TransformedTargetRegressor(
        regressor=HistGradientBoostingRegressor(max_iter=200, random_state=42),
        func=np.log1p,
        inverse_func=np.expm1,
    )


def _train_val_test_split(
    X: np.ndarray, y: np.ndarray, random_state: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=random_state
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def _train_val_test_split_with_phase(
    X: np.ndarray,
    y: np.ndarray,
    phases: np.ndarray,
    random_state: int = 42,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    X_train, X_temp, y_train, y_temp, ph_train, ph_temp = train_test_split(
        X, y, phases, test_size=0.4, random_state=random_state
    )
    X_val, X_test, y_val, y_test, ph_val, ph_test = train_test_split(
        X_temp, y_temp, ph_temp, test_size=0.5, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test, ph_train, ph_val, ph_test


def run_training(
    target_kind: str = DEFAULT_TARGET_KIND,
    *,
    feature_policy: str = "baseline",
    report_path: Path | None = None,
    random_state: int = 42,
) -> None:
    """
    Full training + report. ``feature_policy='strict_planning'`` uses only planning-safe features
    (see ``feature_registry``). Splits use ``random_state`` for reproducibility.
    """
    if target_kind not in TARGET_KIND_CHOICES:
        raise ValueError(f"target_kind must be one of {TARGET_KIND_CHOICES}, got {target_kind!r}")
    if feature_policy not in FEATURE_POLICY_CHOICES:
        raise ValueError(f"feature_policy must be one of {FEATURE_POLICY_CHOICES}, got {feature_policy!r}")

    out_path = resolve_report_path(target_kind, feature_policy, report_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Training run: target_kind=%s feature_policy=%s report=%s random_state=%s",
        target_kind,
        feature_policy,
        out_path,
        random_state,
    )

    df = load_and_join(
        eligibility_columns=KEPT_ELIGIBILITY,
        site_footprint_columns=KEPT_SITE_FOOTPRINT,
        design_columns=KEPT_DESIGN,
        arm_intervention_columns=KEPT_ARM_INTERVENTION,
        design_outcomes_columns=KEPT_DESIGN_OUTCOMES,
    )

    completed = df[df["overall_status"] == "COMPLETED"].copy()
    completed["phase"] = completed["phase"].astype(str)
    phase_counts = completed["phase"].value_counts()

    logger.info(
        "COMPLETED trials cohort: %s rows (counts before per-model target NaN/negative filter)",
        f"{len(completed):,}",
    )

    t_label, t_formula = describe_target_kind(target_kind)
    lines: list[str] = []
    lines.append("REGRESSION TARGET (y) — all RMSE/MAE below are in days for this target")
    lines.append(f"  target_kind: {target_kind}")
    lines.append(f"  label: {t_label}")
    lines.append(f"  definition: {t_formula}")
    lines.append("")
    lines.append("FEATURE POLICY")
    lines.append(f"  feature_policy: {feature_policy}")
    if feature_policy == "strict_planning":
        lines.append(
            "  Planning-safe features only (no start_year, no site-footprint / post-launch operational fields)."
        )
    else:
        lines.append("  Full baseline feature set (historical production configuration).")
    lines.append("")
    lines.append(
        "HistGradientBoostingRegressor: dedicated models for PHASE1, PHASE2, PHASE3; "
        "early joint (PHASE1+PHASE1/PHASE2+PHASE2) for PHASE1/PHASE2 rows; "
        "late joint (PHASE2+PHASE2/PHASE3+PHASE3) for PHASE2/PHASE3 rows."
    )
    lines.append("No StandardScaler; numeric NaNs preserved for HGBR. No phase one-hot inside each cohort.")
    lines.append("")
    lines.append("Trial counts in loaded cohort (COMPLETED, by phase label):")
    for ph in PHASE_REPORT_ORDER:
        n = int(phase_counts.get(ph, 0))
        lines.append(f"  {ph}: n={n:,}")
    lines.append("")

    prep_kw = dict(
        eligibility_columns=KEPT_ELIGIBILITY,
        eligibility_criteria_text_columns=KEPT_ELIGIBILITY_CRITERIA_TEXT,
        site_footprint_columns=KEPT_SITE_FOOTPRINT,
        design_columns=KEPT_DESIGN,
        arm_intervention_columns=KEPT_ARM_INTERVENTION,
        design_outcomes_columns=KEPT_DESIGN_OUTCOMES,
        encode_phase=False,
        target_kind=target_kind,
        policy=feature_policy,
    )

    # phase_label -> (test_n, test_r2, model_description)
    summary: dict[str, tuple[int, float, str]] = {}
    early_by_phase: dict[str, dict] = {}
    late_by_phase: dict[str, dict] = {}

    # --- Dedicated single-phase models (PHASE1, PHASE2, PHASE3) ---
    for phase in PHASE_SINGLE_MODELS:
        df_p = completed[completed["phase"] == phase].copy()
        n_phase = len(df_p)
        lines.append("=" * 50)
        lines.append(f"MODEL dedicated {phase}  [y={target_kind}]")
        lines.append("=" * 50)
        lines.append(f"  Rows before target filter: {n_phase:,}")

        if n_phase < 30:
            lines.append("  Skipped: not enough rows for a stable train/val/test split.")
            lines.append("")
            continue

        X, y, _, _ = prepare_features(df_p, **prep_kw)
        n_xy = len(y)
        logger.info(
            "Dedicated %s: before_target=%s after_target=%s",
            phase,
            f"{n_phase:,}",
            f"{n_xy:,}",
        )
        lines.append(f"  Rows after target filter (finite, non-negative y): {n_xy:,}")

        if n_xy < 30:
            lines.append("  Skipped: too few rows after target (missing/invalid/negative filter).")
            lines.append("")
            continue

        X_train, X_val, X_test, y_train, y_val, y_test = _train_val_test_split(
            X, y, random_state=random_state
        )
        lines.append(
            f"  Split: train={len(y_train):,}  val={len(y_val):,}  test={len(y_test):,}"
        )

        model = _new_regressor()
        model.fit(X_train, y_train)

        for split_name, X_s, y_s in (
            ("train", X_train, y_train),
            ("val", X_val, y_val),
            ("test", X_test, y_test),
        ):
            m = evaluate_sklearn_split(split_name, model, X_s, y_s)
            lines.append(metrics_report_line(m))

        test_m = evaluate_sklearn_split("test", model, X_test, y_test)
        summary[phase] = (len(y_test), test_m["r2"], f"dedicated {phase}-only")
        lines.append("")

    # --- Early joint: train on PHASE1 + PHASE1/PHASE2 + PHASE2; report subset PHASE1/PHASE2 on test ---
    df_early = completed[completed["phase"].isin(EARLY_JOINT_PHASES)].copy()
    lines.append("=" * 50)
    lines.append(
        f"MODEL early joint  (PHASE1 + PHASE1/PHASE2 + PHASE2) [y={target_kind}]"
    )
    lines.append("=" * 50)
    lines.append(f"  Rows before target filter: {len(df_early):,}")
    if len(df_early) < 30:
        lines.append("  Skipped: cohort too small.")
        lines.append("")
    else:
        X_e, y_e, ph_e, _ = prepare_features(df_early, **prep_kw)
        logger.info(
            "Early joint pool: before_target=%s after_target=%s",
            f"{len(df_early):,}",
            f"{len(y_e):,}",
        )
        lines.append(f"  Rows after target filter (finite, non-negative y): {len(y_e):,}")
        if len(y_e) < 30:
            lines.append("  Skipped: too few rows after target (missing/invalid/negative filter).")
            lines.append("")
        else:
            X_tr, X_va, X_te_e, y_tr, y_va, y_te_e, _, _, ph_te_e = _train_val_test_split_with_phase(
                X_e, y_e, ph_e, random_state=random_state
            )
            lines.append(
                f"  Split: train={len(y_tr):,}  val={len(y_va):,}  test={len(y_te_e):,}  (all early-pool phases)"
            )
            m_early = _new_regressor()
            m_early.fit(X_tr, y_tr)
            for split_name, X_s, y_s in (
                ("train", X_tr, y_tr),
                ("val", X_va, y_va),
                ("test", X_te_e, y_te_e),
            ):
                m = evaluate_sklearn_split(split_name, m_early, X_s, y_s)
                lines.append(metrics_report_line(m))
            early_by_phase = _joint_test_metrics_by_phase(
                m_early,
                X_te_e,
                y_te_e,
                ph_te_e,
                ("PHASE1", "PHASE1/PHASE2", "PHASE2"),
            )
            for ph in ("PHASE1", "PHASE1/PHASE2", "PHASE2"):
                if ph not in early_by_phase:
                    lines.append(f"  test ({ph} subset): too few rows for R².")
                    continue
                d = early_by_phase[ph]
                lines.append(joint_subset_report_line(ph, d))
            if "PHASE1/PHASE2" in early_by_phase:
                d12 = early_by_phase["PHASE1/PHASE2"]
                summary["PHASE1/PHASE2"] = (
                    d12["n"],
                    d12["r2"],
                    "early joint (PHASE1+PHASE1/PHASE2+PHASE2)",
                )
            lines.append("")

    # --- Late joint: train on PHASE2 + PHASE2/PHASE3 + PHASE3; report subset PHASE2/PHASE3 on test ---
    df_late = completed[completed["phase"].isin(LATE_JOINT_PHASES)].copy()
    lines.append("=" * 50)
    lines.append(
        f"MODEL late joint  (PHASE2 + PHASE2/PHASE3 + PHASE3) [y={target_kind}]"
    )
    lines.append("=" * 50)
    lines.append(f"  Rows before target filter: {len(df_late):,}")
    if len(df_late) < 30:
        lines.append("  Skipped: cohort too small.")
        lines.append("")
    else:
        X_l, y_l, ph_l, _ = prepare_features(df_late, **prep_kw)
        logger.info(
            "Late joint pool: before_target=%s after_target=%s",
            f"{len(df_late):,}",
            f"{len(y_l):,}",
        )
        lines.append(f"  Rows after target filter (finite, non-negative y): {len(y_l):,}")
        if len(y_l) < 30:
            lines.append("  Skipped: too few rows after target (missing/invalid/negative filter).")
            lines.append("")
        else:
            X_tr, X_va, X_te_l, y_tr, y_va, y_te_l, _, _, ph_te_l = _train_val_test_split_with_phase(
                X_l, y_l, ph_l, random_state=random_state
            )
            lines.append(
                f"  Split: train={len(y_tr):,}  val={len(y_va):,}  test={len(y_te_l):,}  (all late-pool phases)"
            )
            m_late = _new_regressor()
            m_late.fit(X_tr, y_tr)
            for split_name, X_s, y_s in (
                ("train", X_tr, y_tr),
                ("val", X_va, y_va),
                ("test", X_te_l, y_te_l),
            ):
                m = evaluate_sklearn_split(split_name, m_late, X_s, y_s)
                lines.append(metrics_report_line(m))
            late_by_phase = _joint_test_metrics_by_phase(
                m_late,
                X_te_l,
                y_te_l,
                ph_te_l,
                ("PHASE2", "PHASE2/PHASE3", "PHASE3"),
            )
            for ph in ("PHASE2", "PHASE2/PHASE3", "PHASE3"):
                if ph not in late_by_phase:
                    lines.append(f"  test ({ph} subset): too few rows for R².")
                    continue
                d = late_by_phase[ph]
                lines.append(joint_subset_report_line(ph, d))
            if "PHASE2/PHASE3" in late_by_phase:
                d23 = late_by_phase["PHASE2/PHASE3"]
                summary["PHASE2/PHASE3"] = (
                    d23["n"],
                    d23["r2"],
                    "late joint (PHASE2+PHASE2/PHASE3+PHASE3)",
                )
            lines.append("")

    # --- Baseline: old approach (dedicated model trained only on mixed-phase rows) ---
    lines.append("=" * 50)
    lines.append(
        f"BASELINE — dedicated model on mixed-phase rows only (previous approach) [y={target_kind}]"
    )
    lines.append("=" * 50)
    for mix_phase in ("PHASE1/PHASE2", "PHASE2/PHASE3"):
        df_m = completed[completed["phase"] == mix_phase].copy()
        n_m = len(df_m)
        lines.append(f"  {mix_phase} cohort rows before target filter: {n_m:,}")
        if n_m < 30:
            lines.append(f"    Skipped: too few rows.")
            continue
        X_m, y_m, _, _ = prepare_features(df_m, **prep_kw)
        logger.info(
            "Mixed-phase %s: before_target=%s after_target=%s",
            mix_phase,
            f"{n_m:,}",
            f"{len(y_m):,}",
        )
        lines.append(f"    Rows after target filter: {len(y_m):,}")
        if len(y_m) < 30:
            lines.append(f"    Skipped after target filter: too few rows.")
            continue
        X_tr, X_va, X_te, y_tr, y_va, y_te = _train_val_test_split(
            X_m, y_m, random_state=random_state
        )
        base = _new_regressor()
        base.fit(X_tr, y_tr)
        te = evaluate_sklearn_split("test", base, X_te, y_te)
        lines.append(mixed_cohort_test_line(len(y_te), te))
    lines.append("")
    lines.append(
        "Interpretation: baseline R² uses a test split drawn only from the mixed-phase cohort; "
        "joint-model R² for that label uses only mixed-phase rows in the joint pool's test fold "
        "(different test composition). Compare qualitatively."
    )
    lines.append("")

    lines.append("=" * 50)
    lines.append(
        f"TABLE — JOINT MODELS, ALL PHASE SUBSETS (that phase's rows in the joint test fold) [y={target_kind}]"
    )
    lines.append("Phase\tWhich model\tTest n\tR²\tRMSE (days)\tMAE (days)")
    joint_comparison_order: list[tuple[str, str]] = [
        ("PHASE1", "JOINT_EARLY"),
        ("PHASE1/PHASE2", "JOINT_EARLY"),
        ("PHASE2", "JOINT_EARLY"),
        ("PHASE2", "JOINT_LATE"),
        ("PHASE2/PHASE3", "JOINT_LATE"),
        ("PHASE3", "JOINT_LATE"),
    ]
    for ph, tag in joint_comparison_order:
        src = early_by_phase if tag == "JOINT_EARLY" else late_by_phase
        if ph not in src:
            continue
        d = src[ph]
        lines.append(
            f"{ph}\t{tag}\t{d['n']:,}\t{d['r2']:.4f}\t{round(d['rmse'])}\t{round(d['mae'])}"
        )
    lines.append("")

    lines.append("=" * 50)
    lines.append(
        f"TABLE — BEST JOINT PER LABEL (highest test R² when both joints cover the label) [y={target_kind}]"
    )
    lines.append("Phase\tWhich model\tTest n\tR²\tRMSE (days)\tMAE (days)")
    for ph in PHASE_REPORT_ORDER:
        cands: list[tuple[str, dict]] = []
        if ph in early_by_phase:
            cands.append(("JOINT_EARLY", early_by_phase[ph]))
        if ph in late_by_phase:
            cands.append(("JOINT_LATE", late_by_phase[ph]))
        if not cands:
            continue
        tag, d = max(cands, key=lambda x: x[1]["r2"])
        lines.append(
            f"{ph}\t{tag}\t{d['n']:,}\t{d['r2']:.4f}\t{round(d['rmse'])}\t{round(d['mae'])}"
        )
    lines.append("")

    lines.append("=" * 50)
    lines.append(f"SUMMARY — TEST R² BY ROW PHASE (model used for that label) [y={target_kind}]")
    lines.append("=" * 50)
    for ph in PHASE_REPORT_ORDER:
        if ph in summary:
            n_t, r2, desc = summary[ph]
            lines.append(f"  {ph}: n={n_t:,}  R²={r2:.4f}  — {desc}")
        else:
            lines.append(f"  {ph}: n/a  (skipped or insufficient test rows)")
    lines.append("=" * 50)
    lines.append(
        f"End of report — target_kind={target_kind!r} feature_policy={feature_policy!r}"
    )
    lines.append("=" * 50)

    report = "\n".join(lines)
    print(report)
    out_path.write_text(report)
    logger.info(
        "Wrote %s (target_kind=%s feature_policy=%s)",
        out_path,
        target_kind,
        feature_policy,
    )


def main(
    target_kind: str = DEFAULT_TARGET_KIND,
    *,
    feature_policy: str = "baseline",
    report_path: Path | None = None,
    random_state: int = 42,
) -> None:
    """Programmatic entry (same defaults as CLI)."""
    run_training(
        target_kind,
        feature_policy=feature_policy,
        report_path=report_path,
        random_state=random_state,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train per-phase duration regression models.")
    parser.add_argument(
        "--target",
        dest="target_kind",
        choices=list(TARGET_KIND_CHOICES),
        default=DEFAULT_TARGET_KIND,
        help="Regression target: primary_completion (default, matches preprocess duration_days), "
        "post_primary_completion, or total_completion (see targets.py).",
    )
    parser.add_argument(
        "--feature-policy",
        dest="feature_policy",
        choices=list(FEATURE_POLICY_CHOICES),
        default="baseline",
        help="baseline (full features) or strict_planning (planning-safe registry only).",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Output report path (default: regression_report.txt for baseline primary, else auto-named).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for train/val/test splits (default: 42).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    _args = _parse_args()
    run_training(
        _args.target_kind,
        feature_policy=_args.feature_policy,
        report_path=_args.report,
        random_state=_args.random_state,
    )
