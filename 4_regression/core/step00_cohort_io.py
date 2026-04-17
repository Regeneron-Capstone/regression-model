"""Step 00: Load and assemble the modeling cohort from cleaned clinical trial tables."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CLEAN_DATA = PROJECT_ROOT / "clean_data"
RAW_DATA = PROJECT_ROOT / "raw_data"
CONDITION_FEATURES = PROJECT_ROOT / "2_condition_mapping" / "output" / "stage3_nct_features.csv"


def _parse_time_frame_days(tf: str) -> float | None:
    """Parse time_frame string to days. Returns None if unparseable."""
    if pd.isna(tf) or not isinstance(tf, str) or not tf.strip():
        return None
    tf = tf.strip().lower()
    m = re.search(r"(\d+(?:\.\d+)?)\s*(day|week|month|year)s?", tf)
    if not m:
        return None
    val = float(m.group(1))
    unit = m.group(2)
    if unit == "day":
        return val
    if unit == "week":
        return val * 7
    if unit == "month":
        return val * 30.44
    if unit == "year":
        return val * 365.25
    return None


def _has_endpoint_keywords(text: str, keywords: list[str]) -> bool:
    if pd.isna(text) or not isinstance(text, str):
        return False
    t = text.lower()
    return any(k in t for k in keywords)


CCSR_COLS = [
    "ccsr_slot1", "ccsr_slot2", "ccsr_slot3",
    "ccsr_domain",
    "has_ccsr",
    "metastatic_flag", "relapsed_refractory_flag",
    "pediatric_flag", "adult_flag", "biomarker_flag",
    "tier_b_only_flag",
]
CCSR_FLAG_COLS = [
    "has_ccsr", "metastatic_flag", "relapsed_refractory_flag",
    "pediatric_flag", "adult_flag", "biomarker_flag", "tier_b_only_flag",
]


def load_and_join(
    eligibility_columns: list[str] | None = None,
    site_footprint_columns: list[str] | None = None,
    design_columns: list[str] | None = None,
    arm_intervention_columns: list[str] | None = None,
    design_outcomes_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Load clean studies, sponsors, and condition features; join on nct_id.
    If eligibility_columns is provided, join eligibilities table for those columns.

    Disease features come from stage3_nct_features.csv (CCSR-based condition mapping),
    joined LEFT on nct_id.  Trials with no mapped condition get has_ccsr=0 and null slots.
    """
    import logging
    logger = logging.getLogger(__name__)

    studies = pd.read_csv(CLEAN_DATA / "studies.csv", low_memory=False)
    sponsors = pd.read_csv(CLEAN_DATA / "sponsors.csv", low_memory=False)

    # Restrict to COMPLETED trials only (actual duration known)
    studies = studies[studies["overall_status"] == "COMPLETED"].copy()
    n_before_join = len(studies)
    logger.info("Cohort before condition join (COMPLETED): %s", f"{n_before_join:,}")

    # Aggregate sponsors: count per nct_id
    sponsor_counts = sponsors.groupby("nct_id").size().reset_index(name="n_sponsors")
    df = studies.merge(sponsor_counts, on="nct_id", how="left")
    df["n_sponsors"] = df["n_sponsors"].fillna(0).astype(int)

    # ── Join CCSR condition features (replaces categorized_output.csv) ────────
    cond_feat = pd.read_csv(CONDITION_FEATURES, low_memory=False)
    cols_to_join = ["nct_id"] + [c for c in CCSR_COLS if c in cond_feat.columns]
    df = df.merge(cond_feat[cols_to_join], on="nct_id", how="left")

    # Defaults for unmatched trials
    for col in CCSR_FLAG_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)

    # ── Diagnostics ──────────────────────────────────────────────────────────
    n_after_join = len(df)
    n_matched = df["nct_id"].isin(cond_feat["nct_id"]).sum()
    has_ccsr_n = df["has_ccsr"].sum() if "has_ccsr" in df.columns else 0
    slot1_n = df["ccsr_slot1"].notna().sum() if "ccsr_slot1" in df.columns else 0

    logger.info("Cohort after condition join: %s (row count unchanged: %s)",
                f"{n_after_join:,}", n_after_join == n_before_join)
    logger.info("Matched nct_ids from stage3_nct_features: %s / %s",
                f"{n_matched:,}", f"{n_before_join:,}")
    logger.info("has_ccsr=1: %s (%.1f%%)", f"{has_ccsr_n:,}",
                100 * has_ccsr_n / max(n_after_join, 1))
    logger.info("ccsr_slot1 non-null: %s (%.1f%%)", f"{slot1_n:,}",
                100 * slot1_n / max(n_after_join, 1))
    if "ccsr_domain" in df.columns:
        top_domain = df["ccsr_domain"].value_counts().head(10)
        logger.info("Top 10 ccsr_domain:\n%s", top_domain.to_string())
    if "ccsr_slot1" in df.columns:
        top_slot1 = df["ccsr_slot1"].value_counts().head(10)
        logger.info("Top 10 ccsr_slot1:\n%s", top_slot1.to_string())

    # category column: derive from ccsr_domain for downstream compatibility
    # (replaces the old categorized_output.csv "category" feature)
    df["category"] = df["ccsr_domain"].fillna("Other_Unclassified")

    # Join downcase_mesh_term from browse_conditions (first per trial)
    bc_path = RAW_DATA / "browse_conditions.csv"
    if bc_path.exists():
        bc = pd.read_csv(bc_path, low_memory=False)
        mesh_col = "downcase_mesh_term" if "downcase_mesh_term" in bc.columns else "mesh_term"
        if mesh_col in bc.columns:
            mesh_agg = bc.groupby("nct_id")[mesh_col].first().reset_index()
            mesh_agg.columns = ["nct_id", "downcase_mesh_term"]
            df = df.merge(mesh_agg, on="nct_id", how="left")
            df["downcase_mesh_term"] = df["downcase_mesh_term"].fillna("unknown")

    # Join intervention_type from interventions (mode per trial)
    int_path = RAW_DATA / "interventions.csv"
    if int_path.exists():
        interventions = pd.read_csv(int_path, low_memory=False)
        if "intervention_type" in interventions.columns:
            int_agg = (
                interventions.groupby("nct_id")["intervention_type"]
                .agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0])
                .reset_index()
            )
            df = df.merge(int_agg, on="nct_id", how="left")
            df["intervention_type"] = df["intervention_type"].fillna("UNKNOWN")

    # Join eligibilities (first row per nct_id)
    elig_path = RAW_DATA / "eligibilities.csv"
    if elig_path.exists() and eligibility_columns:
        elig = pd.read_csv(elig_path, low_memory=False)
        cols_to_join = ["nct_id"] + [c for c in eligibility_columns if c in elig.columns]
        if len(cols_to_join) > 1:
            elig_agg = elig[cols_to_join].groupby("nct_id").first().reset_index()
            df = df.merge(elig_agg, on="nct_id", how="left")

    # Join site footprint (calculated_values, facilities, countries)
    if site_footprint_columns:
        cv_path = RAW_DATA / "calculated_values.csv"
        if cv_path.exists():
            cv = pd.read_csv(cv_path, low_memory=False)
            cv_cols = ["nct_id", "number_of_facilities", "has_us_facility", "has_single_facility"]
            cv_cols = [c for c in cv_cols if c in cv.columns]
            cv_agg = cv[cv_cols].groupby("nct_id").first().reset_index()
            df = df.merge(cv_agg, on="nct_id", how="left")

        countries_path = RAW_DATA / "countries.csv"
        if countries_path.exists():
            countries = pd.read_csv(countries_path, low_memory=False)
            # Exclude removed countries (removed=True means no longer associated)
            if "removed" in countries.columns:
                countries_active = countries[~countries["removed"].fillna(False).astype(bool)]
            else:
                countries_active = countries
            n_countries = countries_active.groupby("nct_id").size().reset_index(name="number_of_countries")
            df = df.merge(n_countries, on="nct_id", how="left")
            # US-only: 1 if exactly 1 country and it's US
            if "name" in countries.columns:
                us_only = (
                    countries_active.groupby("nct_id")["name"]
                    .apply(lambda x: 1 if (len(x) == 1 and "united states" in str(x.iloc[0]).lower()) else 0)
                    .reset_index(name="us_only")
                )
                df = df.merge(us_only, on="nct_id", how="left")

        fac_path = RAW_DATA / "facilities.csv"
        if fac_path.exists() and "number_of_us_states" in site_footprint_columns:
            fac = pd.read_csv(fac_path, low_memory=False)
            us_fac = fac[fac["country"].str.upper().str.contains("UNITED STATES", na=False)]
            n_us_states = us_fac.groupby("nct_id")["state"].nunique().reset_index(name="number_of_us_states")
            df = df.merge(n_us_states, on="nct_id", how="left")

        # Derived: facility_density = number_of_facilities / enrollment
        if (
            "facility_density" in site_footprint_columns
            and "number_of_facilities" in df.columns
            and "enrollment" in df.columns
        ):
            enroll = pd.to_numeric(df["enrollment"], errors="coerce").fillna(1)
            df["facility_density"] = df["number_of_facilities"].fillna(0) / enroll.replace(0, 1)

    # Join designs (one row per nct_id)
    if design_columns:
        designs_path = RAW_DATA / "designs.csv"
        if designs_path.exists():
            designs = pd.read_csv(designs_path, low_memory=False)
            design_cols = [
                "nct_id",
                "allocation",
                "intervention_model",
                "primary_purpose",
                "masking",
                "subject_masked",
                "caregiver_masked",
                "investigator_masked",
                "outcomes_assessor_masked",
            ]
            design_cols = [c for c in design_cols if c in designs.columns]
            design_agg = designs[design_cols].groupby("nct_id").first().reset_index()
            df = df.merge(design_agg, on="nct_id", how="left")

            # Derived: randomized (1 if RANDOMIZED)
            if "randomized" in design_columns and "allocation" in df.columns:
                df["randomized"] = (df["allocation"].str.upper() == "RANDOMIZED").astype(int)

            # Derived: masking_depth_score (NONE=0, SINGLE=1, DOUBLE=2, TRIPLE=3, QUADRUPLE=4)
            if "masking_depth_score" in design_columns and "masking" in df.columns:
                mask_map = {"NONE": 0, "SINGLE": 1, "DOUBLE": 2, "TRIPLE": 3, "QUADRUPLE": 4}
                df["masking_depth_score"] = df["masking"].str.upper().map(mask_map).fillna(0)
                # Add role flags: +0.25 per masked role (max 1 extra)
                for role in [
                    "subject_masked",
                    "caregiver_masked",
                    "investigator_masked",
                    "outcomes_assessor_masked",
                ]:
                    if role in df.columns:
                        df["masking_depth_score"] += df[role].apply(
                            lambda x: 0.25 if x in (True, "true", "True", 1) else 0
                        )

            # Derived: design_complexity_composite (randomized + multi-arm + normalized masking)
            if "design_complexity_composite" in design_columns:
                r = (df["allocation"].str.upper() == "RANDOMIZED").astype(int) if "allocation" in df.columns else 0
                m = df["masking_depth_score"].fillna(0) if "masking_depth_score" in df.columns else 0
                if "masking_depth_score" not in df.columns and "masking" in df.columns:
                    mask_map = {"NONE": 0, "SINGLE": 1, "DOUBLE": 2, "TRIPLE": 3, "QUADRUPLE": 4}
                    m = df["masking"].str.upper().map(mask_map).fillna(0)
                arms = pd.to_numeric(df["number_of_arms"], errors="coerce").fillna(1)
                multi = (arms > 1).astype(int)
                df["design_complexity_composite"] = r + multi + (m / 5)

    # Join arm/intervention complexity (interventions, design_groups, browse_interventions)
    if arm_intervention_columns:
        int_path = RAW_DATA / "interventions.csv"
        if int_path.exists():
            interventions = pd.read_csv(int_path, low_memory=False)
            if "intervention_type" in interventions.columns:
                n_int = interventions.groupby("nct_id").size().reset_index(name="number_of_interventions")
                df = df.merge(n_int, on="nct_id", how="left")
                n_types = (
                    interventions.groupby("nct_id")["intervention_type"]
                    .nunique()
                    .reset_index(name="intervention_type_diversity")
                )
                df = df.merge(n_types, on="nct_id", how="left")
                df["mono_therapy"] = (df["intervention_type_diversity"].fillna(0) == 1).astype(int)

        dg_path = RAW_DATA / "design_groups.csv"
        if dg_path.exists():
            dg = pd.read_csv(dg_path, low_memory=False)
            if "group_type" in dg.columns:
                dg["_gt"] = dg["group_type"].fillna("").astype(str).str.upper()
                dg["_title"] = dg.get("title", pd.Series([""] * len(dg))).fillna("").astype(str).str.upper()
                dg["_combined"] = dg["_gt"] + " " + dg["_title"]
                has_placebo = (
                    dg.groupby("nct_id")["_combined"]
                    .apply(lambda x: 1 if x.str.contains("PLACEBO", na=False).any() else 0)
                    .reset_index(name="has_placebo")
                )
                df = df.merge(has_placebo, on="nct_id", how="left")
                has_ac = (
                    dg.groupby("nct_id")["_combined"]
                    .apply(
                        lambda x: 1
                        if x.str.contains("ACTIVE.COMPARATOR|ACTIVE_COMPARATOR|COMPARATOR", na=False, regex=True).any()
                        else 0
                    )
                    .reset_index(name="has_active_comparator")
                )
                df = df.merge(has_ac, on="nct_id", how="left")

        bi_path = RAW_DATA / "browse_interventions.csv"
        if bi_path.exists():
            bi = pd.read_csv(bi_path, low_memory=False)
            mesh_col = "downcase_mesh_term" if "downcase_mesh_term" in bi.columns else "mesh_term"
            if mesh_col in bi.columns:
                n_mesh = bi.groupby("nct_id")[mesh_col].nunique().reset_index(name="n_mesh_intervention_terms")
                df = df.merge(n_mesh, on="nct_id", how="left")

    # Join design_outcomes (per-trial aggregates)
    if design_outcomes_columns:
        do_path = RAW_DATA / "design_outcomes.csv"
        if do_path.exists():
            nct_ids = set(df["nct_id"].unique())
            usecols = ["nct_id", "outcome_type", "measure", "time_frame"]
            if "description" in pd.read_csv(do_path, nrows=0).columns:
                usecols.append("description")
            chunks = []
            for chunk in pd.read_csv(do_path, chunksize=200_000, low_memory=False, usecols=usecols):
                chunk = chunk[chunk["nct_id"].isin(nct_ids)]
                if len(chunk) > 0:
                    chunks.append(chunk)
            do = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=usecols + ["_tf_days"])
            if len(do) == 0:
                do = None
            else:
                do["_tf_days"] = do["time_frame"].apply(_parse_time_frame_days)
                SURVIVAL_KW = ["survival", "os", "pfs", "dfs", "overall survival", "progression-free survival"]
                SAFETY_KW = ["safety", "adverse", "ae", "sae", "toxicity", "tolerability"]
                meas = do["measure"] if "measure" in do.columns else pd.Series([""] * len(do))
                desc = do["description"] if "description" in do.columns else pd.Series([""] * len(do))
                do["_has_survival"] = meas.apply(lambda x: _has_endpoint_keywords(x, SURVIVAL_KW)) | desc.apply(
                    lambda x: _has_endpoint_keywords(x, SURVIVAL_KW)
                )
                do["_has_safety"] = meas.apply(lambda x: _has_endpoint_keywords(x, SAFETY_KW)) | desc.apply(
                    lambda x: _has_endpoint_keywords(x, SAFETY_KW)
                )
                n_outcomes = do.groupby("nct_id").size().reset_index(name="n_outcomes")
                max_tf = do.groupby("nct_id")["_tf_days"].max().reset_index(name="max_planned_followup_days")
                agg = n_outcomes.merge(max_tf, on="nct_id", how="left")
                if "outcome_type" in do.columns:
                    n_prim = (
                        do.groupby("nct_id")["outcome_type"]
                        .apply(lambda x: (x.fillna("").str.upper() == "PRIMARY").sum())
                        .reset_index(name="n_primary_outcomes")
                    )
                    n_sec = (
                        do.groupby("nct_id")["outcome_type"]
                        .apply(lambda x: (x.fillna("").str.upper() == "SECONDARY").sum())
                        .reset_index(name="n_secondary_outcomes")
                    )
                    agg = agg.merge(n_prim, on="nct_id", how="left").merge(n_sec, on="nct_id", how="left")
                else:
                    agg["n_primary_outcomes"] = 0
                    agg["n_secondary_outcomes"] = 0
                has_surv = do.groupby("nct_id")["_has_survival"].max().reset_index(name="has_survival_endpoint")
                has_safe = do.groupby("nct_id")["_has_safety"].max().reset_index(name="has_safety_endpoint")
                agg = agg.merge(has_surv, on="nct_id", how="left").merge(has_safe, on="nct_id", how="left")
                agg["endpoint_complexity_score"] = (
                    agg["n_outcomes"].fillna(0) * 0.5
                    + agg["n_primary_outcomes"].fillna(0) * 0.3
                    + agg["n_secondary_outcomes"].fillna(0) * 0.2
                    + agg["has_survival_endpoint"].fillna(0) * 2
                    + agg["has_safety_endpoint"].fillna(0) * 1
                )
                df = df.merge(agg, on="nct_id", how="left")

    return df
