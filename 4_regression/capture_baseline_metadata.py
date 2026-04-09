"""
Emit baseline modeling metadata: row counts per split, feature counts, feature column names.

Reads the same data and uses the same prepare_features / split helpers as train_regression.py.
Does not train models or change any saved model artifacts.

Usage (after preprocess + train_regression, or standalone — loads data fresh):
  python 4_regression/capture_baseline_metadata.py
  python 4_regression/capture_baseline_metadata.py --output results/baseline_metadata.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from targets import DEFAULT_TARGET_KIND  # noqa: E402

from cohort_columns import (  # noqa: E402
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
)
from cohort_io import load_and_join  # noqa: E402
from targets import TARGET_DURATION_COLUMN  # noqa: E402
from train_regression import (  # noqa: E402
    _train_val_test_split,
    _train_val_test_split_with_phase,
    prepare_features,
)

TARGET_COLUMN = TARGET_DURATION_COLUMN

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _feature_names_from_artifacts(art: dict, *, encode_phase: bool) -> list[str]:
    """Column order matches prepare_features X = hstack(...)."""
    names: list[str] = []
    pe = art.get("phase_encoder")
    if encode_phase and pe is not None:
        names.extend(pe.get_feature_names_out(["phase"]))
    names.extend(art["cat_encoder"].get_feature_names_out(["category"]))
    me = art.get("mesh_encoder")
    if me is not None:
        names.extend(me.get_feature_names_out(["mesh_trimmed"]))
    ie = art.get("int_encoder")
    if ie is not None:
        names.extend(ie.get_feature_names_out(["intervention_trimmed"]))
    names.extend(art.get("elig_feature_names", []))
    names.extend(art.get("criteria_text_feature_names", []))
    names.extend(art.get("site_feature_names", []))
    names.extend(art.get("design_feature_names", []))
    names.extend(art.get("do_feature_names", []))
    names.extend(art.get("arm_feature_names", []))
    names.extend(
        art.get("numeric_feature_names", ["enrollment", "n_sponsors", "number_of_arms", "start_year"])
    )
    return names


def _split_record(
    X: np.ndarray,
    y: np.ndarray,
    *,
    with_phase: bool = False,
    phases: np.ndarray | None = None,
    random_state: int = 42,
) -> dict:
    if with_phase and phases is not None:
        X_tr, X_va, X_te, y_tr, y_va, y_te, ph_tr, ph_va, ph_te = _train_val_test_split_with_phase(
            X, y, phases, random_state=random_state
        )
        test_by_phase: dict[str, int] = {}
        for ph in PHASE_REPORT_ORDER:
            test_by_phase[ph] = int((ph_te == ph).sum())
        return {
            "train": int(len(y_tr)),
            "val": int(len(y_va)),
            "test": int(len(y_te)),
            "test_rows_by_phase": test_by_phase,
        }
    X_tr, X_va, X_te, y_tr, y_va, y_te = _train_val_test_split(X, y, random_state=random_state)
    return {
        "train": int(len(y_tr)),
        "val": int(len(y_va)),
        "test": int(len(y_te)),
    }


def build_metadata(random_state: int = 42) -> dict:
    prep_kw = dict(
        eligibility_columns=KEPT_ELIGIBILITY,
        eligibility_criteria_text_columns=KEPT_ELIGIBILITY_CRITERIA_TEXT,
        site_footprint_columns=KEPT_SITE_FOOTPRINT,
        design_columns=KEPT_DESIGN,
        arm_intervention_columns=KEPT_ARM_INTERVENTION,
        design_outcomes_columns=KEPT_DESIGN_OUTCOMES,
        encode_phase=False,
        target_kind=DEFAULT_TARGET_KIND,
    )

    logger.info("Loading joined data...")
    df = load_and_join(
        eligibility_columns=KEPT_ELIGIBILITY,
        site_footprint_columns=KEPT_SITE_FOOTPRINT,
        design_columns=KEPT_DESIGN,
        arm_intervention_columns=KEPT_ARM_INTERVENTION,
        design_outcomes_columns=KEPT_DESIGN_OUTCOMES,
    )
    completed = df[df["overall_status"] == "COMPLETED"].copy()
    completed["phase"] = completed["phase"].astype(str)
    phase_counts_loaded = {ph: int(completed["phase"].eq(ph).sum()) for ph in PHASE_REPORT_ORDER}

    models: dict = {}

    for phase in PHASE_SINGLE_MODELS:
        df_p = completed[completed["phase"] == phase].copy()
        n_before = len(df_p)
        key = f"dedicated_{phase}"
        if n_before < 30:
            models[key] = {"skipped": True, "reason": "n<30 before features", "rows_before_dropna_target": n_before}
            continue
        X, y, _, art = prepare_features(df_p, **prep_kw)
        n_after = len(y)
        if n_after < 30:
            models[key] = {
                "skipped": True,
                "reason": "n<30 after dropna target",
                "rows_before_dropna_target": n_before,
                "rows_after_dropna_target": n_after,
            }
            continue
        fnames = _feature_names_from_artifacts(art, encode_phase=False)
        splits = _split_record(X, y, random_state=random_state)
        models[key] = {
            "skipped": False,
            "rows_before_dropna_target": n_before,
            "rows_after_dropna_target": n_after,
            "target_column": TARGET_COLUMN,
            "n_features": len(fnames),
            "feature_names": fnames,
            "splits_60_20_20": splits,
        }

    # Joint early
    df_e = completed[completed["phase"].isin(EARLY_JOINT_PHASES)].copy()
    key_e = "joint_early_PHASE1_PHASE1_2_PHASE2"
    if len(df_e) < 30:
        models[key_e] = {"skipped": True, "reason": "pool n<30", "rows_before_dropna_target": len(df_e)}
    else:
        X, y, ph, art = prepare_features(df_e, **prep_kw)
        if len(y) < 30:
            models[key_e] = {"skipped": True, "reason": "after dropna n<30", "rows_after_dropna_target": len(y)}
        else:
            fnames = _feature_names_from_artifacts(art, encode_phase=False)
            splits = _split_record(X, y, with_phase=True, phases=ph, random_state=random_state)
            models[key_e] = {
                "skipped": False,
                "rows_before_dropna_target": len(df_e),
                "rows_after_dropna_target": len(y),
                "phases_in_pool": sorted(EARLY_JOINT_PHASES),
                "target_column": TARGET_COLUMN,
                "n_features": len(fnames),
                "feature_names": fnames,
                "splits_60_20_20": splits,
            }

    # Joint late
    df_l = completed[completed["phase"].isin(LATE_JOINT_PHASES)].copy()
    key_l = "joint_late_PHASE2_PHASE2_3_PHASE3"
    if len(df_l) < 30:
        models[key_l] = {"skipped": True, "reason": "pool n<30", "rows_before_dropna_target": len(df_l)}
    else:
        X, y, ph, art = prepare_features(df_l, **prep_kw)
        if len(y) < 30:
            models[key_l] = {"skipped": True, "reason": "after dropna n<30", "rows_after_dropna_target": len(y)}
        else:
            fnames = _feature_names_from_artifacts(art, encode_phase=False)
            splits = _split_record(X, y, with_phase=True, phases=ph, random_state=random_state)
            models[key_l] = {
                "skipped": False,
                "rows_before_dropna_target": len(df_l),
                "rows_after_dropna_target": len(y),
                "phases_in_pool": sorted(LATE_JOINT_PHASES),
                "target_column": TARGET_COLUMN,
                "n_features": len(fnames),
                "feature_names": fnames,
                "splits_60_20_20": splits,
            }

    # Reference: same feature layout as dedicated models (for quick diff — use first non-skipped dedicated)
    reference_feature_names: list[str] | None = None
    reference_n_features: int | None = None
    for phase in PHASE_SINGLE_MODELS:
        m = models.get(f"dedicated_{phase}")
        if m and not m.get("skipped"):
            reference_feature_names = m["feature_names"]
            reference_n_features = m["n_features"]
            break

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "split_rule": "train_test_split test_size=0.4 then 0.5 on holdout → 60/20/20 train/val/test",
        "random_state": random_state,
        "completed_trials_total": int(len(completed)),
        "completed_trials_by_phase_label": phase_counts_loaded,
        "target_kind": DEFAULT_TARGET_KIND,
        "target_column_primary_completion": TARGET_COLUMN,
        "models": models,
        "reference_feature_layout": {
            "description": "First non-skipped dedicated single-phase model (PHASE1/2/3); "
            "n_features may differ slightly between cohorts due to OneHotEncoder categories.",
            "n_features": reference_n_features,
            "feature_names": reference_feature_names,
        },
        "deviation_script_note": "5_deviation/deviation_analysis.py (or baseline_deviation.py wrapper) trains only "
        "dedicated PHASE1/2/3 for primary_completion; same prepare_features columns as dedicated_* above.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Write baseline_metadata.json for regression baseline freeze.")
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR / "baseline_metadata.json",
        help="Output JSON path (default: results/baseline_metadata.json)",
    )
    args = parser.parse_args()

    meta = build_metadata()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(meta, indent=2))
    logger.info("Wrote %s", args.output)


if __name__ == "__main__":
    main()
