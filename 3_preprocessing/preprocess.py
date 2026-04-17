"""
Preprocess studies and sponsors with filtering criteria.
Saves to clean_data folder.
"""
import logging
from pathlib import Path

import pandas as pd

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
RAW_DATA = PROJECT_ROOT / "raw_data"
CLEAN_DATA = PROJECT_ROOT / "clean_data"
OUTPUT_DIR = CLEAN_DATA
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Filtering criteria
ALLOWED_PHASES = {"PHASE1", "PHASE2", "PHASE3", "PHASE1/PHASE2", "PHASE2/PHASE3"}
# duration_days = primary_completion_date − start_date (days)
MIN_DURATION_DAYS = 14  # drop bottom outliers / implausibly short windows
MAX_DURATION_DAYS = 3650  # 10 years — cap top outliers

CRITERIA_TEXT_FEATURE_COLUMNS = [
    "eligibility_criteria_char_len",
    "eligibility_n_inclusion_tildes",
    "eligibility_n_exclusion_tildes",
    "eligibility_has_burden_procedure",
]

BURDEN_KEYWORDS = [
    "biopsy",
    "mri",
    "ecg",
    "ekg",
    "washout",
    "endoscopy",
    "colonoscopy",
    "bronchoscopy",
    "lumbar puncture",
    "spinal tap",
    "pet scan",
    "ct scan",
    "cardiac catheterization",
]


def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    logger.info("Loading raw data")
    studies = pd.read_csv(RAW_DATA / "studies.csv", low_memory=False)
    sponsors = pd.read_csv(RAW_DATA / "sponsors.csv", low_memory=False)
    logger.info("Studies: %s rows, Sponsors: %s rows", f"{len(studies):,}", f"{len(sponsors):,}")
    return studies, sponsors


def filter_sponsors(sponsors: pd.DataFrame) -> pd.DataFrame:
    """Filter to industry sponsors only."""
    filtered = sponsors[sponsors["agency_class"] == "INDUSTRY"].copy()
    logger.info("Sponsors (agency_class=INDUSTRY): %s rows", f"{len(filtered):,}")
    return filtered


def filter_studies(
    studies: pd.DataFrame,
    industry_nct_ids: set[str],
) -> pd.DataFrame:
    """
    Filter studies:
    - Exclude overall_status = WITHDRAWN
    - study_type = INTERVENTIONAL
    - phase in PHASE1, PHASE2, PHASE3, PHASE1/PHASE2, PHASE2/PHASE3
    - nct_id must have at least one industry sponsor
    """
    # Exclude WITHDRAWN
    filtered = studies[studies["overall_status"] != "WITHDRAWN"].copy()
    logger.info("After excluding WITHDRAWN: %s rows", f"{len(filtered):,}")

    # study_type = INTERVENTIONAL
    filtered = filtered[filtered["study_type"] == "INTERVENTIONAL"]
    logger.info("After study_type=INTERVENTIONAL: %s rows", f"{len(filtered):,}")

    # phase in allowed phases
    filtered = filtered[filtered["phase"].isin(ALLOWED_PHASES)]
    logger.info("After phase filter: %s rows", f"{len(filtered):,}")

    # Must have industry sponsor
    filtered = filtered[filtered["nct_id"].isin(industry_nct_ids)]
    logger.info("After industry sponsor filter: %s rows", f"{len(filtered):,}")

    return filtered


def count_inclusion_tildes(text: str) -> int:
    if not isinstance(text, str) or not text.strip():
        return 0
    if "Inclusion Criteria:" not in text:
        return 0
    try:
        before_excl = text.split("Exclusion Criteria:")[0]
        inclusion_part = before_excl.split("Inclusion Criteria:")[1]
        return inclusion_part.count("~")
    except (IndexError, ValueError):
        return 0


def count_exclusion_tildes(text: str) -> int:
    if not isinstance(text, str) or not text.strip():
        return 0
    if "Exclusion Criteria:" not in text:
        return 0
    exclusion_part = text.split("Exclusion Criteria:")[1]
    return exclusion_part.count("~")


def has_burden_keyword(text: str) -> bool:
    if not isinstance(text, str) or not text.strip():
        return False
    t = text.lower()
    return any(kw in t for kw in BURDEN_KEYWORDS)


def compute_criteria_features_for_eligibilities(elig: pd.DataFrame) -> pd.DataFrame:
    """
    Input: columns nct_id, criteria (one row per nct_id recommended).
    Output: nct_id + CRITERIA_TEXT_FEATURE_COLUMNS (int).
    """
    if "nct_id" not in elig.columns or "criteria" not in elig.columns:
        raise ValueError("elig must contain nct_id and criteria")
    t = elig["criteria"].fillna("").astype(str)
    out = pd.DataFrame(
        {
            "nct_id": elig["nct_id"].values,
            "eligibility_criteria_char_len": t.str.len().astype("int64"),
            "eligibility_n_inclusion_tildes": t.map(count_inclusion_tildes).astype("int64"),
            "eligibility_n_exclusion_tildes": t.map(count_exclusion_tildes).astype("int64"),
            "eligibility_has_burden_procedure": t.map(lambda x: int(has_burden_keyword(x))).astype("int64"),
        }
    )
    return out


def merge_eligibility_criteria_text_features(filtered_studies: pd.DataFrame) -> pd.DataFrame:
    """Join parsed criteria text features from raw eligibilities onto studies (one row per nct_id)."""
    path = RAW_DATA / "eligibilities.csv"
    if not path.exists():
        logger.warning("eligibilities.csv not found; criteria text features set to 0")
        out = filtered_studies.copy()
        for c in CRITERIA_TEXT_FEATURE_COLUMNS:
            out[c] = 0
        return out

    elig = pd.read_csv(path, usecols=["nct_id", "criteria"], low_memory=False)
    elig = elig[elig["nct_id"].isin(filtered_studies["nct_id"])]
    elig = elig.groupby("nct_id", as_index=False).first()
    feats = compute_criteria_features_for_eligibilities(elig)
    out = filtered_studies.merge(feats, on="nct_id", how="left")
    for c in CRITERIA_TEXT_FEATURE_COLUMNS:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype("int64")
    logger.info(
        "Merged eligibility criteria text features onto %s trials",
        f"{len(out):,}",
    )
    return out


def compute_enrollment_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute enrollment statistics by phase."""
    # enrollment may be stored as string or numeric
    df = df.copy()
    df["enrollment"] = pd.to_numeric(df["enrollment"], errors="coerce")

    rows = []
    for phase in sorted(df["phase"].unique()):
        phase_df = df[df["phase"] == phase]
        total = len(phase_df)
        with_enrollment = phase_df["enrollment"].notna().sum()
        missing = total - with_enrollment
        pct_missing = round(missing / total * 100, 2) if total > 0 else 0

        enrollments = phase_df["enrollment"].dropna()
        if len(enrollments) > 0:
            mean_enrollment = round(enrollments.mean(), 1)
            median_enrollment = round(enrollments.median(), 1)
            q1 = enrollments.quantile(0.25)
            q3 = enrollments.quantile(0.75)
            iqr = round(q3 - q1, 1)
        else:
            mean_enrollment = median_enrollment = q1 = q3 = iqr = None

        rows.append(
            {
                "phase": phase,
                "total_trials": total,
                "trials_with_enrollment": int(with_enrollment),
                "missing_enrollment": int(missing),
                "pct_missing": pct_missing,
                "mean_enrollment": mean_enrollment,
                "median_enrollment": median_enrollment,
                "q1_enrollment": q1,
                "q3_enrollment": q3,
                "iqr_enrollment": iqr,
            }
        )

    return pd.DataFrame(rows)


def save_and_report(
    filtered_studies: pd.DataFrame,
    filtered_sponsors: pd.DataFrame,
    enrollment_stats: pd.DataFrame,
) -> None:
    """Save to 0_data/clean_data and write summary report."""
    # Save CSVs
    studies_path = OUTPUT_DIR / "studies.csv"
    sponsors_path = OUTPUT_DIR / "sponsors.csv"
    filtered_studies.to_csv(studies_path, index=False)
    filtered_sponsors.to_csv(sponsors_path, index=False)
    logger.info("Saved %s", studies_path)
    logger.info("Saved %s", sponsors_path)

    # Save enrollment stats
    stats_path = OUTPUT_DIR / "enrollment_stats_by_phase.csv"
    enrollment_stats.to_csv(stats_path, index=False)
    logger.info("Saved %s", stats_path)

    # Summary report
    lines = []
    lines.append("=" * 60)
    lines.append("PREPROCESSING SUMMARY")
    lines.append("=" * 60)
    lines.append(f"Total trials (filtered studies): {len(filtered_studies):,}")
    lines.append(f"Total sponsors (industry only): {len(filtered_sponsors):,}")
    lines.append("")
    lines.append("Eligibility criteria text features (from raw eligibilities.criteria):")
    lines.append(f"  {', '.join(CRITERIA_TEXT_FEATURE_COLUMNS)}")
    lines.append("")
    lines.append("-" * 40)
    lines.append("TOTAL TRIALS BY PHASE")
    lines.append("-" * 40)
    for _, row in enrollment_stats.iterrows():
        lines.append(f"  {row['phase']}: {row['total_trials']:,}")
    lines.append("")
    lines.append("-" * 40)
    lines.append("ENROLLMENT STATISTICS BY PHASE")
    lines.append("-" * 40)
    for _, row in enrollment_stats.iterrows():
        lines.append(f"  {row['phase']}:")
        lines.append(f"    total_trials: {row['total_trials']:,}")
        lines.append(f"    trials_with_enrollment: {row['trials_with_enrollment']:,}")
        lines.append(f"    missing_enrollment: {row['missing_enrollment']:,} ({row['pct_missing']}%)")
        lines.append(f"    mean_enrollment: {row['mean_enrollment']}")
        lines.append(f"    median_enrollment: {row['median_enrollment']}")
        lines.append(f"    q1_enrollment: {row['q1_enrollment']}")
        lines.append(f"    q3_enrollment: {row['q3_enrollment']}")
        lines.append(f"    iqr_enrollment: {row['iqr_enrollment']}")
        lines.append("")
    lines.append("=" * 60)

    report = "\n".join(lines)
    print(report)
    (OUTPUT_DIR / "preprocessing_summary.txt").write_text(report)
    logger.info("Saved preprocessing_summary.txt")


def main() -> None:
    studies, sponsors = load_raw_data()

    # Filter sponsors to industry only
    filtered_sponsors = filter_sponsors(sponsors)
    industry_nct_ids = set(filtered_sponsors["nct_id"].unique())

    # Filter studies
    filtered_studies = filter_studies(studies, industry_nct_ids)

    # Parse dates and filter: drop nulls, keep 1980 <= date <= 2027
    filtered_studies = filtered_studies.copy()
    start = pd.to_datetime(filtered_studies["start_date"], errors="coerce")
    primary_end = pd.to_datetime(filtered_studies["primary_completion_date"], errors="coerce")

    # Drop nulls in start_date or primary_completion_date
    mask_valid = start.notna() & primary_end.notna()
    filtered_studies = filtered_studies[mask_valid].copy()
    start = start[mask_valid]
    primary_end = primary_end[mask_valid]
    logger.info("After dropping null dates: %s rows", f"{len(filtered_studies):,}")

    # Keep only dates in range 1980-2027 (inclusive)
    date_min = pd.Timestamp("1980-01-01")
    date_max = pd.Timestamp("2027-12-31")
    mask_in_range = (start >= date_min) & (start <= date_max) & (primary_end >= date_min) & (primary_end <= date_max)
    filtered_studies = filtered_studies[mask_in_range].copy()
    start = start[mask_in_range]
    primary_end = primary_end[mask_in_range]
    logger.info("After date range 1980-2027: %s rows", f"{len(filtered_studies):,}")

    # Duration: Primary Completion - Start (days)
    filtered_studies["duration_days"] = (primary_end - start).dt.days

    # Drop rows with negative duration (primary completion before start)
    before_drop = len(filtered_studies)
    filtered_studies = filtered_studies[filtered_studies["duration_days"] >= 0].copy()
    dropped = before_drop - len(filtered_studies)
    if dropped > 0:
        logger.info("After dropping negative duration: %s rows (removed %s)", f"{len(filtered_studies):,}", dropped)

    before_band = len(filtered_studies)
    dur = filtered_studies["duration_days"]
    mask_duration_band = (dur >= MIN_DURATION_DAYS) & (dur <= MAX_DURATION_DAYS)
    filtered_studies = filtered_studies[mask_duration_band].copy()
    dropped_band = before_band - len(filtered_studies)
    logger.info(
        "After duration band [%s, %s] days: %s rows (removed %s)",
        MIN_DURATION_DAYS,
        MAX_DURATION_DAYS,
        f"{len(filtered_studies):,}",
        f"{dropped_band:,}",
    )

    # Add modeling columns
    filtered_studies["is_completed"] = filtered_studies["overall_status"] == "COMPLETED"

    # Filter sponsors to only those for our filtered studies
    filtered_sponsors = filtered_sponsors[
        filtered_sponsors["nct_id"].isin(filtered_studies["nct_id"])
    ].copy()

    # Eligibility criteria text features → columns on studies for regression
    filtered_studies = merge_eligibility_criteria_text_features(filtered_studies)

    # Enrollment stats by phase
    enrollment_stats = compute_enrollment_stats(filtered_studies)

    save_and_report(filtered_studies, filtered_sponsors, enrollment_stats)
    logger.info("Done. Outputs in %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
