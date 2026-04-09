"""
Shared column bundles and phase routing for completed-trial modeling.

Imported by ``train_regression``, deviation analysis, metadata capture, and other
pipelines so they do not depend on each other's modules for these constants.
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_PREPROC = _PROJECT_ROOT / "3_preprocessing"
if str(_PREPROC) not in sys.path:
    sys.path.insert(0, str(_PREPROC))
from eligibility_criteria_features import CRITERIA_TEXT_FEATURE_COLUMNS  # noqa: E402

# Best-performing columns from ablation studies (see MODEL.md)
KEPT_ELIGIBILITY = ["gender", "minimum_age", "maximum_age", "adult", "child", "older_adult"]
KEPT_ELIGIBILITY_CRITERIA_TEXT = list(CRITERIA_TEXT_FEATURE_COLUMNS)
KEPT_SITE_FOOTPRINT = ["number_of_facilities", "number_of_countries", "us_only", "has_single_facility"]
KEPT_DESIGN = [
    "randomized",
    "intervention_model",
    "masking_depth_score",
    "primary_purpose",
    "design_complexity_composite",
]
KEPT_ARM_INTERVENTION = [
    "number_of_interventions",
    "intervention_type_diversity",
    "mono_therapy",
    "has_placebo",
    "has_active_comparator",
    "n_mesh_intervention_terms",
]
KEPT_DESIGN_OUTCOMES = [
    "max_planned_followup_days",
    "n_primary_outcomes",
    "n_secondary_outcomes",
    "n_outcomes",
    "has_survival_endpoint",
    "has_safety_endpoint",
    "endpoint_complexity_score",
]

PHASE_SINGLE_MODELS: tuple[str, ...] = ("PHASE1", "PHASE2", "PHASE3")
# Alias for scripts that expect this name
PHASES_WITH_DEDICATED_MODELS = PHASE_SINGLE_MODELS

EARLY_JOINT_PHASES = frozenset({"PHASE1", "PHASE1/PHASE2", "PHASE2"})
LATE_JOINT_PHASES = frozenset({"PHASE2", "PHASE2/PHASE3", "PHASE3"})

PHASE_REPORT_ORDER: tuple[str, ...] = (
    "PHASE1",
    "PHASE1/PHASE2",
    "PHASE2",
    "PHASE2/PHASE3",
    "PHASE3",
)


def default_feature_prep_kw(*, policy: str, target_kind: str) -> dict:
    """Keyword args for ``prepare_features`` / ``assemble_feature_matrix`` (encode_phase=False)."""
    return dict(
        eligibility_columns=KEPT_ELIGIBILITY,
        eligibility_criteria_text_columns=KEPT_ELIGIBILITY_CRITERIA_TEXT,
        site_footprint_columns=KEPT_SITE_FOOTPRINT,
        design_columns=KEPT_DESIGN,
        arm_intervention_columns=KEPT_ARM_INTERVENTION,
        design_outcomes_columns=KEPT_DESIGN_OUTCOMES,
        encode_phase=False,
        policy=policy,
        target_kind=target_kind,
    )
