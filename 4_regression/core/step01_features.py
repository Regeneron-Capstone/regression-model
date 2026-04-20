"""Step 01: Construct the feature matrix from the cohort using domain-driven feature groups."""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

_SCRIPT_DIR = Path(__file__).resolve().parent
_REGRESSION_DIR = _SCRIPT_DIR.parent
if str(_REGRESSION_DIR) not in sys.path:
    sys.path.insert(0, str(_REGRESSION_DIR))

from feature_registry import get_feature_policy, validate_no_leakage
from step02_targets import (
    DEFAULT_TARGET_KIND,
    MODEL_TARGET_INTERNAL_COL,
    resolve_target_series,
)

logger = logging.getLogger(__name__)

PolicyName = Literal["baseline", "strict_planning"]


def _filter_columns_for_policy(columns: list[str] | None, policy: PolicyName) -> list[str] | None:
    if columns is None or policy == "baseline":
        return columns
    forbidden = get_feature_policy("strict_planning").forbidden
    return [c for c in columns if c not in forbidden]


def add_start_year_column(df: pd.DataFrame) -> pd.DataFrame:
    """Derive ``start_year`` from ``start_date`` (same as legacy prepare_features)."""
    out = df.copy()
    if "start_date" in out.columns:
        out["start_year"] = pd.to_datetime(out["start_date"], errors="coerce").dt.year
    else:
        out["start_year"] = np.nan
    out["start_year"] = pd.to_numeric(out["start_year"], errors="coerce")
    return out


def coerce_core_numerics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["enrollment", "number_of_arms"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        else:
            out[col] = np.nan
    return out


def attach_target_by_kind(df: pd.DataFrame, target_kind: str) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Attach ``MODEL_TARGET_INTERNAL_COL`` from ``resolve_target_series``, drop rows with
    missing or negative spans, log drops.
    """
    n_in = len(df)
    series = resolve_target_series(df, target_kind)
    out = df.copy()
    out[MODEL_TARGET_INTERNAL_COL] = series
    out = out.dropna(subset=[MODEL_TARGET_INTERNAL_COL])
    dropped_na = n_in - len(out)
    if dropped_na:
        logger.info(
            "Target %s: dropped %s / %s rows with missing target",
            target_kind,
            f"{dropped_na:,}",
            f"{n_in:,}",
        )
    neg_mask = out[MODEL_TARGET_INTERNAL_COL] < 0
    n_neg = int(neg_mask.sum())
    if n_neg:
        out = out.loc[~neg_mask].copy()
        logger.info(
            "Target %s: dropped %s rows with negative span",
            target_kind,
            f"{n_neg:,}",
        )
    y = out[MODEL_TARGET_INTERNAL_COL].to_numpy(dtype=np.float64)
    return out, y


def build_phase_block(df: pd.DataFrame, *, encode_phase: bool) -> tuple[list[np.ndarray], object | None]:
    if encode_phase:
        phase_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        phase_encoded = phase_encoder.fit_transform(df[["phase"]])
        return [phase_encoded], phase_encoder
    return [], None


def build_category_block(df: pd.DataFrame) -> tuple[np.ndarray, OneHotEncoder]:
    cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    cat_encoded = cat_encoder.fit_transform(df[["category"]])
    return cat_encoded, cat_encoder


def build_mesh_block(
    df: pd.DataFrame,
    *,
    top_mesh: list[str] | None = None,
) -> tuple[list[np.ndarray], OneHotEncoder | None, list[str]]:
    if "downcase_mesh_term" not in df.columns:
        return [], None, []
    if top_mesh is None:
        top_mesh = df["downcase_mesh_term"].value_counts().head(50).index.tolist()
    dfc = df.copy()
    dfc["mesh_trimmed"] = dfc["downcase_mesh_term"].where(dfc["downcase_mesh_term"].isin(top_mesh), "other")
    mesh_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return [mesh_encoder.fit_transform(dfc[["mesh_trimmed"]])], mesh_encoder, list(top_mesh)


def build_intervention_type_block(
    df: pd.DataFrame,
    *,
    top_intervention_types: list[str] | None = None,
) -> tuple[list[np.ndarray], OneHotEncoder | None, list[str]]:
    if "intervention_type" not in df.columns:
        return [], None, []
    if top_intervention_types is None:
        top_intervention_types = df["intervention_type"].value_counts().head(15).index.tolist()
    dfc = df.copy()
    dfc["intervention_trimmed"] = dfc["intervention_type"].where(
        dfc["intervention_type"].isin(top_intervention_types), "other"
    )
    int_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return [int_encoder.fit_transform(dfc[["intervention_trimmed"]])], int_encoder, list(top_intervention_types)


def build_eligibility_blocks(
    df: pd.DataFrame, eligibility_columns: list[str] | None
) -> tuple[list[np.ndarray], dict, list[str]]:
    elig_parts: list[np.ndarray] = []
    elig_encoders: dict = {}
    elig_feature_names: list[str] = []
    if not eligibility_columns:
        return elig_parts, elig_encoders, elig_feature_names
    for col in eligibility_columns:
        if col not in df.columns:
            continue
        if col == "gender":
            dfc = df.copy()
            dfc["gender_fill"] = dfc["gender"].fillna("ALL").astype(str)
            enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            elig_parts.append(enc.fit_transform(dfc[["gender_fill"]]))
            elig_encoders["gender"] = enc
            elig_feature_names.extend(enc.get_feature_names_out(["gender_fill"]))
        elif col in ("minimum_age", "maximum_age"):
            raw = df[col].astype(str).str.extract(r"(\d+)", expand=False)
            vals = pd.to_numeric(raw, errors="coerce")
            elig_parts.append(np.column_stack([vals.to_numpy(dtype=np.float64, na_value=np.nan)]))
            elig_feature_names.append(col)
        elif col in ("adult", "child", "older_adult"):
            s = df[col]

            def _tri_state(v: object) -> float:
                if pd.isna(v):
                    return np.nan
                return 1.0 if v in (True, "true", "True", "YES", "Yes", 1) else 0.0

            elig_parts.append(np.column_stack([s.map(_tri_state).astype(np.float64).values]))
            elig_feature_names.append(col)
    return elig_parts, elig_encoders, elig_feature_names


def build_criteria_text_blocks(
    df: pd.DataFrame, eligibility_criteria_text_columns: list[str] | None
) -> tuple[list[np.ndarray], list[str]]:
    criteria_parts: list[np.ndarray] = []
    criteria_text_feature_names: list[str] = []
    if not eligibility_criteria_text_columns:
        return criteria_parts, criteria_text_feature_names
    for col in eligibility_criteria_text_columns:
        if col not in df.columns:
            vals = np.zeros(len(df), dtype=float)
        else:
            vals = pd.to_numeric(df[col], errors="coerce").fillna(0).to_numpy(dtype=float)
        criteria_parts.append(np.column_stack([vals]))
        criteria_text_feature_names.append(col)
    return criteria_parts, criteria_text_feature_names


def build_site_footprint_blocks(
    df: pd.DataFrame, site_footprint_columns: list[str] | None
) -> tuple[list[np.ndarray], list[str]]:
    site_parts: list[np.ndarray] = []
    site_feature_names: list[str] = []
    if not site_footprint_columns:
        return site_parts, site_feature_names
    for col in site_footprint_columns:
        if col not in df.columns:
            continue
        if col == "has_single_facility":

            def _sf(x: object) -> float:
                if pd.isna(x):
                    return np.nan
                return 1.0 if x in (True, "true", "True", "YES", "Yes", 1) else 0.0

            vals = df[col].map(_sf).astype(np.float64)
            site_parts.append(np.column_stack([vals.values]))
            site_feature_names.append(col)
        elif col in ("number_of_facilities", "number_of_countries", "number_of_us_states"):
            vals = pd.to_numeric(df[col], errors="coerce")
            site_parts.append(np.column_stack([vals.to_numpy(dtype=np.float64, na_value=np.nan)]))
            site_feature_names.append(col)
        elif col == "us_only":
            vals = pd.to_numeric(df[col], errors="coerce")
            site_parts.append(np.column_stack([vals.to_numpy(dtype=np.float64, na_value=np.nan)]))
            site_feature_names.append(col)
        elif col == "facility_density":
            vals = pd.to_numeric(df[col], errors="coerce")
            site_parts.append(np.column_stack([vals.to_numpy(dtype=np.float64, na_value=np.nan)]))
            site_feature_names.append(col)
    return site_parts, site_feature_names


def build_design_blocks(
    df: pd.DataFrame,
    design_columns: list[str] | None,
    *,
    intervention_model_top: list[str] | None = None,
    primary_purpose_top: list[str] | None = None,
) -> tuple[list[np.ndarray], list[str], dict[str, OneHotEncoder], dict[str, list[str]]]:
    design_parts: list[np.ndarray] = []
    design_feature_names: list[str] = []
    design_ohes: dict[str, OneHotEncoder] = {}
    design_tops: dict[str, list[str]] = {}
    if not design_columns:
        return design_parts, design_feature_names, design_ohes, design_tops
    for col in design_columns:
        if col not in df.columns:
            continue
        if col == "randomized":
            vals = pd.to_numeric(df[col], errors="coerce")
            design_parts.append(np.column_stack([vals.to_numpy(dtype=np.float64, na_value=np.nan)]))
            design_feature_names.append(col)
        elif col == "intervention_model":
            dfc = df.copy()
            dfc["intervention_model_fill"] = dfc["intervention_model"].fillna("UNKNOWN").astype(str)
            top_mod = (
                list(intervention_model_top)
                if intervention_model_top is not None
                else dfc["intervention_model_fill"].value_counts().head(6).index.tolist()
            )
            dfc["intervention_model_trimmed"] = dfc["intervention_model_fill"].where(
                dfc["intervention_model_fill"].isin(top_mod), "other"
            )
            enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            design_parts.append(enc.fit_transform(dfc[["intervention_model_trimmed"]]))
            design_feature_names.extend(enc.get_feature_names_out(["intervention_model_trimmed"]))
            design_ohes["intervention_model"] = enc
            design_tops["intervention_model"] = top_mod
        elif col == "primary_purpose":
            dfc = df.copy()
            dfc["primary_purpose_fill"] = dfc["primary_purpose"].fillna("OTHER").astype(str)
            top_pp = (
                list(primary_purpose_top)
                if primary_purpose_top is not None
                else dfc["primary_purpose_fill"].value_counts().head(6).index.tolist()
            )
            dfc["primary_purpose_trimmed"] = dfc["primary_purpose_fill"].where(
                dfc["primary_purpose_fill"].isin(top_pp), "other"
            )
            enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            design_parts.append(enc.fit_transform(dfc[["primary_purpose_trimmed"]]))
            design_feature_names.extend(enc.get_feature_names_out(["primary_purpose_trimmed"]))
            design_ohes["primary_purpose"] = enc
            design_tops["primary_purpose"] = top_pp
        elif col in ("masking_depth_score", "design_complexity_composite"):
            vals = pd.to_numeric(df[col], errors="coerce")
            design_parts.append(np.column_stack([vals.to_numpy(dtype=np.float64, na_value=np.nan)]))
            design_feature_names.append(col)
    return design_parts, design_feature_names, design_ohes, design_tops


def build_design_outcomes_blocks(
    df: pd.DataFrame, design_outcomes_columns: list[str] | None
) -> tuple[list[np.ndarray], list[str]]:
    do_parts: list[np.ndarray] = []
    do_feature_names: list[str] = []
    if not design_outcomes_columns:
        return do_parts, do_feature_names
    for col in design_outcomes_columns:
        if col not in df.columns:
            continue
        if col in ("has_survival_endpoint", "has_safety_endpoint"):

            def _ep(x: object) -> float:
                if pd.isna(x):
                    return np.nan
                return 1.0 if x in (True, "true", "True", 1) else 0.0

            vals = df[col].map(_ep).astype(np.float64)
            do_parts.append(np.column_stack([vals.values]))
            do_feature_names.append(col)
        else:
            vals = pd.to_numeric(df[col], errors="coerce")
            do_parts.append(np.column_stack([vals.to_numpy(dtype=np.float64, na_value=np.nan)]))
            do_feature_names.append(col)
    return do_parts, do_feature_names


def build_arm_intervention_blocks(
    df: pd.DataFrame, arm_intervention_columns: list[str] | None
) -> tuple[list[np.ndarray], list[str]]:
    arm_parts: list[np.ndarray] = []
    arm_feature_names: list[str] = []
    if not arm_intervention_columns:
        return arm_parts, arm_feature_names
    for col in arm_intervention_columns:
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        arm_parts.append(np.column_stack([vals.to_numpy(dtype=np.float64, na_value=np.nan)]))
        arm_feature_names.append(col)
    return arm_parts, arm_feature_names


def build_numeric_tail(
    df: pd.DataFrame, *, include_start_year: bool
) -> tuple[np.ndarray, list[str]]:
    enroll = pd.to_numeric(df["enrollment"], errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)
    n_spon = pd.to_numeric(df["n_sponsors"], errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)
    n_arms = pd.to_numeric(df["number_of_arms"], errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)
    names = ["enrollment", "n_sponsors", "number_of_arms"]
    if include_start_year:
        sy = pd.to_numeric(df["start_year"], errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)
        X_numeric = np.column_stack([enroll, n_spon, n_arms, sy])
        names = names + ["start_year"]
    else:
        X_numeric = np.column_stack([enroll, n_spon, n_arms])
    return X_numeric, names


def _collect_logical_inputs_for_validation(
    *,
    eligibility_columns: list[str] | None,
    eligibility_criteria_text_columns: list[str] | None,
    site_footprint_columns: list[str] | None,
    design_columns: list[str] | None,
    arm_intervention_columns: list[str] | None,
    design_outcomes_columns: list[str] | None,
    include_start_year: bool,
    include_mesh_term: bool,
) -> list[str]:
    """Logical study-level columns feeding the matrix (excluding one-hot expanded names)."""
    out: list[str] = [
        "phase",
        "category",
        "intervention_type",
        "enrollment",
        "n_sponsors",
        "number_of_arms",
    ]
    if include_mesh_term:
        out.insert(2, "downcase_mesh_term")
    if include_start_year:
        out.append("start_year")
    for group in (
        eligibility_columns,
        eligibility_criteria_text_columns,
        site_footprint_columns,
        design_columns,
        arm_intervention_columns,
        design_outcomes_columns,
    ):
        if group:
            out.extend(group)
    return out


def assemble_feature_matrix(
    df: pd.DataFrame,
    eligibility_columns: list[str] | None = None,
    eligibility_criteria_text_columns: list[str] | None = None,
    site_footprint_columns: list[str] | None = None,
    design_columns: list[str] | None = None,
    arm_intervention_columns: list[str] | None = None,
    design_outcomes_columns: list[str] | None = None,
    *,
    encode_phase: bool = False,
    include_mesh_term: bool = False,
    policy: PolicyName = "baseline",
    target_kind: str = DEFAULT_TARGET_KIND,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Build ``X``, ``y``, phase labels, and encoder artifacts.

    Parameters
    ----------
    policy
        ``baseline`` — same feature sets as historical training (caller lists unchanged).
        ``strict_planning`` — drop any column in ``feature_registry.STRICT_PLANNING_FORBIDDEN``
        from the optional column lists and omit ``start_year`` from the numeric tail.

    target_kind
        Which label to predict: ``primary_completion`` (default), ``post_primary_completion``,
        or ``total_completion`` — see ``targets.resolve_target_series``.

    Note
    ----
    Default ``policy`` is ``baseline`` so production training behavior is unchanged when
    callers use the same arguments as before.
    """
    df = add_start_year_column(df)
    df = coerce_core_numerics(df)

    elig_cols = _filter_columns_for_policy(eligibility_columns, policy)
    crit_cols = _filter_columns_for_policy(eligibility_criteria_text_columns, policy)
    site_cols = _filter_columns_for_policy(site_footprint_columns, policy)
    des_cols = _filter_columns_for_policy(design_columns, policy)
    arm_cols = _filter_columns_for_policy(arm_intervention_columns, policy)
    do_cols = _filter_columns_for_policy(design_outcomes_columns, policy)
    include_start_year = policy == "baseline"

    if policy == "strict_planning":
        logical = _collect_logical_inputs_for_validation(
            eligibility_columns=elig_cols,
            eligibility_criteria_text_columns=crit_cols,
            site_footprint_columns=site_cols,
            design_columns=des_cols,
            arm_intervention_columns=arm_cols,
            design_outcomes_columns=do_cols,
            include_start_year=include_start_year,
            include_mesh_term=include_mesh_term,
        )
        validate_no_leakage(logical, get_feature_policy("strict_planning").forbidden)

    df, y = attach_target_by_kind(df, target_kind)

    phase_blocks, phase_encoder = build_phase_block(df, encode_phase=encode_phase)
    cat_encoded, cat_encoder = build_category_block(df)
    if include_mesh_term:
        mesh_parts, mesh_encoder, mesh_top_terms = build_mesh_block(df)
    else:
        mesh_parts, mesh_encoder, mesh_top_terms = [], None, []
    int_parts, int_encoder, int_top_terms = build_intervention_type_block(df)
    elig_parts, elig_encoders, elig_feature_names = build_eligibility_blocks(df, elig_cols)
    criteria_parts, criteria_text_feature_names = build_criteria_text_blocks(df, crit_cols)
    site_parts, site_feature_names = build_site_footprint_blocks(df, site_cols)
    design_parts, design_feature_names, design_ohes, design_tops = build_design_blocks(df, des_cols)
    do_parts, do_feature_names = build_design_outcomes_blocks(df, do_cols)
    arm_parts, arm_feature_names = build_arm_intervention_blocks(df, arm_cols)
    X_numeric, numeric_feature_names = build_numeric_tail(df, include_start_year=include_start_year)

    X = np.hstack(
        phase_blocks
        + [cat_encoded]
        + mesh_parts
        + int_parts
        + elig_parts
        + criteria_parts
        + site_parts
        + design_parts
        + do_parts
        + arm_parts
        + [X_numeric]
    )
    X = np.asarray(X, dtype=np.float64)
    phases = df["phase"].astype(str).values

    logical_source_columns = sorted(
        set(
            _collect_logical_inputs_for_validation(
                eligibility_columns=elig_cols,
                eligibility_criteria_text_columns=crit_cols,
                site_footprint_columns=site_cols,
                design_columns=des_cols,
                arm_intervention_columns=arm_cols,
                design_outcomes_columns=do_cols,
                include_start_year=include_start_year,
                include_mesh_term=include_mesh_term,
            )
        )
    )

    artifacts = {
        "phase_encoder": phase_encoder,
        "cat_encoder": cat_encoder,
        "include_mesh_term": include_mesh_term,
        "mesh_encoder": mesh_encoder,
        "int_encoder": int_encoder,
        "mesh_top_terms": mesh_top_terms,
        "intervention_type_top_terms": int_top_terms,
        "design_encoders": design_ohes,
        "intervention_model_top_terms": design_tops.get("intervention_model", []),
        "primary_purpose_top_terms": design_tops.get("primary_purpose", []),
        "elig_encoders": elig_encoders,
        "elig_feature_names": elig_feature_names,
        "criteria_text_feature_names": criteria_text_feature_names,
        "site_feature_names": site_feature_names,
        "design_feature_names": design_feature_names,
        "do_feature_names": do_feature_names,
        "arm_feature_names": arm_feature_names,
        "numeric_feature_names": numeric_feature_names,
        "feature_policy": policy,
        "logical_source_columns": logical_source_columns,
        "target_kind": target_kind,
        "model_target_internal_col": MODEL_TARGET_INTERNAL_COL,
        "encode_phase": encode_phase,
    }
    if "nct_id" in df.columns:
        artifacts["nct_ids"] = df["nct_id"].astype(str).to_numpy()
    return X, y, phases, artifacts


def feature_matrix_column_names(artifacts: dict) -> list[str]:
    """
    Names for columns of ``X`` from ``assemble_feature_matrix``, in hstack order
    (must match ``transform_feature_matrix``).
    """
    names: list[str] = []
    phase_encoder = artifacts.get("phase_encoder")
    if bool(artifacts.get("encode_phase")) and phase_encoder is not None:
        names.extend(list(phase_encoder.get_feature_names_out(["phase"])))
    cat_encoder = artifacts.get("cat_encoder")
    if cat_encoder is not None:
        names.extend(list(cat_encoder.get_feature_names_out(["category"])))
    if artifacts.get("include_mesh_term") and artifacts.get("mesh_encoder") is not None:
        mesh_enc = artifacts["mesh_encoder"]
        names.extend(list(mesh_enc.get_feature_names_out(["mesh_trimmed"])))
    int_encoder = artifacts.get("int_encoder")
    if int_encoder is not None:
        names.extend(list(int_encoder.get_feature_names_out(["intervention_trimmed"])))
    names.extend(list(artifacts.get("elig_feature_names") or []))
    names.extend(list(artifacts.get("criteria_text_feature_names") or []))
    names.extend(list(artifacts.get("site_feature_names") or []))
    names.extend(list(artifacts.get("design_feature_names") or []))
    names.extend(list(artifacts.get("do_feature_names") or []))
    names.extend(list(artifacts.get("arm_feature_names") or []))
    names.extend(list(artifacts.get("numeric_feature_names") or []))
    return names


def _transform_eligibility_blocks(
    df: pd.DataFrame,
    eligibility_columns: list[str] | None,
    elig_encoders: dict,
) -> list[np.ndarray]:
    elig_parts: list[np.ndarray] = []
    if not eligibility_columns:
        return elig_parts
    for col in eligibility_columns:
        if col not in df.columns:
            continue
        if col == "gender":
            enc = elig_encoders.get("gender")
            if enc is None:
                continue
            dfc = df.copy()
            dfc["gender_fill"] = dfc["gender"].fillna("ALL").astype(str)
            elig_parts.append(enc.transform(dfc[["gender_fill"]]))
        elif col in ("minimum_age", "maximum_age"):
            raw = df[col].astype(str).str.extract(r"(\d+)", expand=False)
            vals = pd.to_numeric(raw, errors="coerce")
            elig_parts.append(np.column_stack([vals.to_numpy(dtype=np.float64, na_value=np.nan)]))
        elif col in ("adult", "child", "older_adult"):
            s = df[col]

            def _tri_state(v: object) -> float:
                if pd.isna(v):
                    return np.nan
                return 1.0 if v in (True, "true", "True", "YES", "Yes", 1) else 0.0

            elig_parts.append(np.column_stack([s.map(_tri_state).astype(np.float64).values]))
    return elig_parts


def _transform_design_blocks(
    df: pd.DataFrame,
    design_columns: list[str] | None,
    artifacts: dict,
) -> list[np.ndarray]:
    design_parts: list[np.ndarray] = []
    if not design_columns:
        return design_parts
    ohes: dict = artifacts.get("design_encoders") or {}
    im_top: list[str] = list(artifacts.get("intervention_model_top_terms") or [])
    pp_top: list[str] = list(artifacts.get("primary_purpose_top_terms") or [])
    for col in design_columns:
        if col not in df.columns:
            continue
        if col == "randomized":
            vals = pd.to_numeric(df[col], errors="coerce")
            design_parts.append(np.column_stack([vals.to_numpy(dtype=np.float64, na_value=np.nan)]))
        elif col == "intervention_model":
            enc = ohes.get("intervention_model")
            if enc is None:
                continue
            dfc = df.copy()
            dfc["intervention_model_fill"] = dfc["intervention_model"].fillna("UNKNOWN").astype(str)
            dfc["intervention_model_trimmed"] = dfc["intervention_model_fill"].where(
                dfc["intervention_model_fill"].isin(im_top), "other"
            )
            design_parts.append(enc.transform(dfc[["intervention_model_trimmed"]]))
        elif col == "primary_purpose":
            enc = ohes.get("primary_purpose")
            if enc is None:
                continue
            dfc = df.copy()
            dfc["primary_purpose_fill"] = dfc["primary_purpose"].fillna("OTHER").astype(str)
            dfc["primary_purpose_trimmed"] = dfc["primary_purpose_fill"].where(
                dfc["primary_purpose_fill"].isin(pp_top), "other"
            )
            design_parts.append(enc.transform(dfc[["primary_purpose_trimmed"]]))
        elif col in ("masking_depth_score", "design_complexity_composite"):
            vals = pd.to_numeric(df[col], errors="coerce")
            design_parts.append(np.column_stack([vals.to_numpy(dtype=np.float64, na_value=np.nan)]))
    return design_parts


def transform_feature_matrix(
    df: pd.DataFrame,
    artifacts: dict,
    eligibility_columns: list[str] | None = None,
    eligibility_criteria_text_columns: list[str] | None = None,
    site_footprint_columns: list[str] | None = None,
    design_columns: list[str] | None = None,
    arm_intervention_columns: list[str] | None = None,
    design_outcomes_columns: list[str] | None = None,
    *,
    encode_phase: bool | None = None,
    include_mesh_term: bool | None = None,
    policy: PolicyName | None = None,
) -> np.ndarray:
    """
    Apply encoders from a prior ``assemble_feature_matrix`` call to new rows (no target attach/drop).

    ``artifacts`` must come from the same feature policy and column configuration as training.
    """
    pol: PolicyName = policy if policy is not None else artifacts["feature_policy"]
    enc_phase = encode_phase if encode_phase is not None else bool(artifacts.get("encode_phase", False))
    if include_mesh_term is not None:
        mesh_on = include_mesh_term
    elif "include_mesh_term" in artifacts:
        mesh_on = bool(artifacts["include_mesh_term"])
    else:
        mesh_on = artifacts.get("mesh_encoder") is not None

    df = add_start_year_column(df.copy())
    df = coerce_core_numerics(df)

    elig_cols = _filter_columns_for_policy(eligibility_columns, pol)
    crit_cols = _filter_columns_for_policy(eligibility_criteria_text_columns, pol)
    site_cols = _filter_columns_for_policy(site_footprint_columns, pol)
    des_cols = _filter_columns_for_policy(design_columns, pol)
    arm_cols = _filter_columns_for_policy(arm_intervention_columns, pol)
    do_cols = _filter_columns_for_policy(design_outcomes_columns, pol)
    include_start_year = pol == "baseline"

    if pol == "strict_planning":
        logical = _collect_logical_inputs_for_validation(
            eligibility_columns=elig_cols,
            eligibility_criteria_text_columns=crit_cols,
            site_footprint_columns=site_cols,
            design_columns=des_cols,
            arm_intervention_columns=arm_cols,
            design_outcomes_columns=do_cols,
            include_start_year=include_start_year,
            include_mesh_term=mesh_on,
        )
        validate_no_leakage(logical, get_feature_policy("strict_planning").forbidden)

    phase_encoder = artifacts.get("phase_encoder")
    if enc_phase and phase_encoder is not None:
        phase_blocks = [phase_encoder.transform(df[["phase"]].astype(str))]
    else:
        phase_blocks = []

    cat_encoder = artifacts["cat_encoder"]
    if "category" not in df.columns:
        df["category"] = "Other_Unclassified"
    cat_encoded = cat_encoder.transform(df[["category"]].astype(str))

    mesh_encoder = artifacts.get("mesh_encoder") if mesh_on else None
    mesh_top_terms: list[str] = list(artifacts.get("mesh_top_terms") or [])
    if mesh_on and mesh_encoder is not None and mesh_top_terms:
        if "downcase_mesh_term" not in df.columns:
            df = df.copy()
            df["downcase_mesh_term"] = "unknown"
        dfc = df.copy()
        dfc["mesh_trimmed"] = dfc["downcase_mesh_term"].where(
            dfc["downcase_mesh_term"].isin(mesh_top_terms), "other"
        )
        mesh_parts = [mesh_encoder.transform(dfc[["mesh_trimmed"]])]
    else:
        mesh_parts = []

    int_encoder = artifacts.get("int_encoder")
    int_top = list(artifacts.get("intervention_type_top_terms") or [])
    if int_encoder is not None and int_top:
        if "intervention_type" not in df.columns:
            df = df.copy()
            df["intervention_type"] = "UNKNOWN"
        dfc = df.copy()
        dfc["intervention_trimmed"] = dfc["intervention_type"].where(
            dfc["intervention_type"].isin(int_top), "other"
        )
        int_parts = [int_encoder.transform(dfc[["intervention_trimmed"]])]
    else:
        int_parts = []

    elig_encoders = artifacts.get("elig_encoders") or {}
    elig_parts = _transform_eligibility_blocks(df, elig_cols, elig_encoders)

    criteria_parts, _ = build_criteria_text_blocks(df, crit_cols)
    site_parts, _ = build_site_footprint_blocks(df, site_cols)
    design_parts = _transform_design_blocks(df, des_cols, artifacts)
    do_parts, _ = build_design_outcomes_blocks(df, do_cols)
    arm_parts, _ = build_arm_intervention_blocks(df, arm_cols)
    X_numeric, _ = build_numeric_tail(df, include_start_year=include_start_year)

    X = np.hstack(
        phase_blocks
        + [cat_encoded]
        + mesh_parts
        + int_parts
        + elig_parts
        + criteria_parts
        + site_parts
        + design_parts
        + do_parts
        + arm_parts
        + [X_numeric]
    )
    return np.asarray(X, dtype=np.float64)
