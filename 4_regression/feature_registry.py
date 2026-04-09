"""
Planning-time vs post-start feature policy registry.

Used to document leakage risk and validate feature lists *before* rewiring
``prepare_features``. Conservative: if a signal is typically finalized or
revised only after the study has opened / accrued operational reality, treat it
as forbidden for strict planning-time prediction.

This module does not encode one-hot expanded names — only logical column names
as used in ``train_regression`` KEPT_* groups and numeric blocks.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, Iterable, Literal

# ---------------------------------------------------------------------------
# Feature groups (planning-safe)
# ---------------------------------------------------------------------------

# Declared in the registry / protocol at ClinicalTrials.gov registration.
PLANNING_SAFE_CORE: FrozenSet[str] = frozenset(
    {
        "phase",
        "enrollment",  # planned or protocol enrollment (field may later be updated — still pre-operational intent)
        "n_sponsors",  # sponsor list is part of the registered record before conduct
        "number_of_arms",
        "category",
        "downcase_mesh_term",
        "intervention_type",
    }
)

PLANNING_SAFE_ELIGIBILITY: FrozenSet[str] = frozenset(
    {
        "gender",
        "minimum_age",
        "maximum_age",
        "adult",
        "child",
        "older_adult",
    }
)

# Derived from eligibility *criteria* text in the protocol (no outcomes).
PLANNING_SAFE_ELIGIBILITY_TEXT: FrozenSet[str] = frozenset(
    {
        "eligibility_criteria_char_len",
        "eligibility_n_inclusion_tildes",
        "eligibility_n_exclusion_tildes",
        "eligibility_has_burden_procedure",
    }
)

# Design / allocation fields from the registered protocol.
PLANNING_SAFE_DESIGN: FrozenSet[str] = frozenset(
    {
        "randomized",
        "intervention_model",
        "masking_depth_score",
        "primary_purpose",
        "design_complexity_composite",
    }
)

# Arm and intervention structure as registered (intervention arms, placebo arms, etc.).
PLANNING_SAFE_ARMS_INTERVENTIONS: FrozenSet[str] = frozenset(
    {
        "number_of_interventions",
        "intervention_type_diversity",
        "mono_therapy",
        "has_placebo",
        "has_active_comparator",
        "n_mesh_intervention_terms",
    }
)

# Outcome *definitions* (counts, planned follow-up windows from outcome time_frame text).
# Conservative caveat: time_frame parsing can still encode post-hoc edits to the record;
# for maximum strictness you may drop this group later.
PLANNING_SAFE_DESIGN_OUTCOMES: FrozenSet[str] = frozenset(
    {
        "max_planned_followup_days",
        "n_primary_outcomes",
        "n_secondary_outcomes",
        "n_outcomes",
        "has_survival_endpoint",
        "has_safety_endpoint",
        "endpoint_complexity_score",
    }
)

STRICT_PLANNING_ALLOWED: FrozenSet[str] = (
    PLANNING_SAFE_CORE
    | PLANNING_SAFE_ELIGIBILITY
    | PLANNING_SAFE_ELIGIBILITY_TEXT
    | PLANNING_SAFE_DESIGN
    | PLANNING_SAFE_ARMS_INTERVENTIONS
    | PLANNING_SAFE_DESIGN_OUTCOMES
)

# ---------------------------------------------------------------------------
# Forbidden for strict planning (with rationale)
# ---------------------------------------------------------------------------

# start_year is derived from ``start_date``. That date is the trial *start* (actual or
# anticipated). Using it as a feature encodes the calendar position of first conduct /
# registration timing relative to outcomes and is not a pure pre-start protocol input
# under a strict “before first patient” reading — also often updated to actual after start.
FORBIDDEN_START_ANCHORED: FrozenSet[str] = frozenset({"start_year"})

# Site footprint reflects listed facilities/countries; in practice lists grow and change as
# sites activate, suspend, or are added after study start. Using counts/flags leaks
# operational execution state after opening.
FORBIDDEN_SITE_FOOTPRINT: FrozenSet[str] = frozenset(
    {
        "number_of_facilities",
        "number_of_countries",
        "us_only",
        "has_single_facility",
        "facility_density",  # derived using enrollment + facilities — still footprint-dependent
        "number_of_us_states",
    }
)

# Targets and obvious post-trial / results leakage (forbid if ever passed as features).
FORBIDDEN_TARGETS_AND_RESULTS: FrozenSet[str] = frozenset(
    {
        "duration_days",
        "target_duration",  # registry field often misused; not used as model feature today
        "actual_duration",  # from calculated_values — purely post hoc
        "months_to_report_results",
        "were_results_reported",
        "number_of_nsae_subjects",
        "number_of_sae_subjects",
    }
)

# Status and completion timestamps encode trial life-cycle after start.
FORBIDDEN_STATUS_AND_DATES: FrozenSet[str] = frozenset(
    {
        "overall_status",
        "last_known_status",
        "start_date",
        "primary_completion_date",
        "completion_date",
        "is_completed",
    }
)

STRICT_PLANNING_FORBIDDEN: FrozenSet[str] = (
    FORBIDDEN_START_ANCHORED
    | FORBIDDEN_SITE_FOOTPRINT
    | FORBIDDEN_TARGETS_AND_RESULTS
    | FORBIDDEN_STATUS_AND_DATES
)

PolicyName = Literal["strict_planning", "leakage_check_only"]


@dataclass(frozen=True)
class FeaturePolicy:
    """Named policy: which logical columns are allowed vs forbidden for planning-time use."""

    name: str
    description: str
    allowed: FrozenSet[str]
    forbidden: FrozenSet[str]


def get_feature_policy(policy_name: PolicyName | str) -> FeaturePolicy:
    """
    Return a feature policy by name.

    - ``strict_planning``: conservative allowlist + explicit forbidden union (for validation).
    - ``leakage_check_only``: forbidden set only (targets, results, status, dates, footprint,
      start_year); allowed left empty — use when you only want ``validate_no_leakage`` and
      will manage allowlists elsewhere.
    """
    if policy_name == "strict_planning":
        return FeaturePolicy(
            name="strict_planning",
            description="Conservative planning-time policy: protocol/design/eligibility/outcome-definition "
            "features only; no start-anchored year, site footprint, targets, or lifecycle fields.",
            allowed=STRICT_PLANNING_ALLOWED,
            forbidden=STRICT_PLANNING_FORBIDDEN,
        )
    if policy_name == "leakage_check_only":
        return FeaturePolicy(
            name="leakage_check_only",
            description="Reject known-leaky columns only; does not enforce full allowlist.",
            allowed=frozenset(),
            forbidden=STRICT_PLANNING_FORBIDDEN,
        )
    raise KeyError(f"Unknown feature policy: {policy_name!r}")


def validate_no_leakage(feature_columns: Iterable[str], forbidden_columns: Iterable[str]) -> None:
    """
    Ensure no forbidden logical feature name appears in ``feature_columns``.

    Raises:
        ValueError: if any intersection with ``forbidden_columns``.
    """
    feats = frozenset(str(c).strip() for c in feature_columns if str(c).strip())
    bad = frozenset(forbidden_columns)
    leaked = feats & bad
    if leaked:
        raise ValueError(
            "Planning-time leakage guard failed: forbidden columns present: "
            f"{sorted(leaked)}"
        )


def validate_strict_planning_feature_set(feature_columns: Iterable[str]) -> None:
    """
    Apply ``strict_planning`` policy: no forbidden columns, and every column must be in the
    conservative allowlist (extras are rejected to prevent silent leakage).
    """
    policy = get_feature_policy("strict_planning")
    validate_no_leakage(feature_columns, policy.forbidden)
    feats = frozenset(str(c).strip() for c in feature_columns if str(c).strip())
    unknown = feats - policy.allowed
    if unknown:
        raise ValueError(
            "Strict planning allowlist: unknown or disallowed columns (not in registry allowed set): "
            f"{sorted(unknown)}"
        )
