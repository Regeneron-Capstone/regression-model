"""
Data exploration for studies.csv: phase, dates, nulls, status, and ongoing studies.
"""
import logging
import warnings
from pathlib import Path

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", message="Could not infer format", module="pandas")
import pandas as pd
import seaborn as sns

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
RAW_DATA = PROJECT_ROOT / "raw_data"
OUTPUT_DIR = PROJECT_ROOT / "2_data_exploration" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Date columns grouped by purpose (from ClinicalTrials.gov schema)
DATE_COLUMNS = {
    "start": ["start_month_year", "start_date_type", "start_date"],
    "primary_completion": [
        "primary_completion_month_year",
        "primary_completion_date_type",
        "primary_completion_date",
    ],
    "completion": ["completion_month_year", "completion_date_type", "completion_date"],
    "verification": ["verification_month_year", "verification_date"],
    "posted": [
        "study_first_posted_date",
        "results_first_posted_date",
        "disposition_first_posted_date",
        "last_update_posted_date",
    ],
    "submitted": [
        "study_first_submitted_date",
        "results_first_submitted_date",
        "disposition_first_submitted_date",
        "last_update_submitted_date",
    ],
    "metadata": ["created_at", "updated_at"],
}


def load_studies() -> pd.DataFrame:
    path = RAW_DATA / "studies.csv"
    logger.info("Loading %s", path)
    df = pd.read_csv(path, low_memory=False)
    logger.info("Loaded %s rows", f"{len(df):,}")
    return df


def analyze_null_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Null counts for all columns, sorted by null count descending."""
    nulls = df.isna().sum()
    null_pct = (nulls / len(df) * 100).round(1)
    result = pd.DataFrame(
        {"null_count": nulls, "null_pct": null_pct},
        index=df.columns,
    ).sort_values("null_count", ascending=False)
    return result


def analyze_date_columns(df: pd.DataFrame) -> dict:
    """Analyze date columns: null counts, ranges, and parseable dates."""
    date_cols = [
        c for c in df.columns if any(d in c.lower() for d in ["date", "year", "month"])
    ]
    results = {}
    for group_name, cols in DATE_COLUMNS.items():
        for col in cols:
            if col not in df.columns:
                continue
            s = df[col].dropna()
            s = s[s.astype(str).str.strip() != ""]
            null_count = df[col].isna().sum()
            empty_count = (df[col].astype(str).str.strip() == "").sum()
            total_null = null_count + empty_count
            results[col] = {
                "group": group_name,
                "null_count": int(total_null),
                "null_pct": round(total_null / len(df) * 100, 1),
                "non_null_count": int(len(s)),
                "sample_values": s.head(3).tolist(),
            }
            # Try to parse as date for range
            if "date" in col.lower() and col not in ("start_date_type", "completion_date_type", "primary_completion_date_type"):
                try:
                    parsed = pd.to_datetime(s, errors="coerce")
                    valid = parsed.dropna()
                    if len(valid) > 0:
                        results[col]["min_date"] = str(valid.min())
                        results[col]["max_date"] = str(valid.max())
                except Exception:
                    pass
    return results


def analyze_phase(df: pd.DataFrame) -> pd.DataFrame:
    """Phase distribution."""
    phase_counts = df["phase"].value_counts(dropna=False)
    phase_pct = (phase_counts / len(df) * 100).round(1)
    return pd.DataFrame(
        {"count": phase_counts, "pct": phase_pct}
    ).sort_values("count", ascending=False)


def analyze_status(df: pd.DataFrame) -> pd.DataFrame:
    """Overall status and last_known_status - key for identifying ongoing studies."""
    status_cols = ["overall_status", "last_known_status"]
    results = {}
    for col in status_cols:
        if col in df.columns:
            counts = df[col].value_counts(dropna=False)
            results[col] = counts
    return results


def ongoing_status_key() -> dict:
    """Map status values to 'ongoing' vs 'completed' vs 'other'."""
    return {
        "ongoing": [
            "RECRUITING",
            "ENROLLING_BY_INVITATION",
            "ACTIVE_NOT_RECRUITING",
            "NOT_YET_RECRUITING",
            "AVAILABLE",
            "SUSPENDED",
            "TEMPORARILY_NOT_AVAILABLE",
            "WITHDRAWN",  # sometimes considered ongoing in some analyses
        ],
        "completed": [
            "COMPLETED",
            "TERMINATED",
            "NO_LONGER_AVAILABLE",
        ],
        "unknown": ["UNKNOWN", ""],
    }


def analyze_date_formats(df: pd.DataFrame) -> str:
    """
    Deep dive on start_date, start_month_year, primary_completion_date:
    format, sample values, value patterns.
    """
    cols = ["start_date", "start_month_year", "primary_completion_date"]
    lines = []
    lines.append("=" * 80)
    lines.append("DATE FORMAT ANALYSIS: start_date, start_month_year, primary_completion_date")
    lines.append("=" * 80)
    lines.append(f"Total rows: {len(df):,}")
    lines.append("")

    for col in cols:
        if col not in df.columns:
            lines.append(f"{col}: NOT FOUND")
            continue

        s = df[col].dropna()
        s = s[s.astype(str).str.strip() != ""]
        null_count = df[col].isna().sum()
        empty = (df[col].astype(str).str.strip() == "").sum()

        lines.append("-" * 60)
        lines.append(f"COLUMN: {col}")
        lines.append("-" * 60)
        lines.append(f"  dtype: {df[col].dtype}")
        lines.append(f"  non-null count: {len(s):,}")
        lines.append(f"  null count: {null_count:,}")
        lines.append(f"  empty string count: {empty:,}")
        lines.append("")

        # Sample values (first 15 unique)
        samples = s.astype(str).unique()[:15].tolist()
        lines.append("  Sample values (first 15 unique):")
        for v in samples:
            lines.append(f"    '{v}'")
        lines.append("")

        # Value format patterns
        str_vals = s.astype(str)
        yyyy_mm_dd = str_vals.str.match(r"^\d{4}-\d{1,2}-\d{1,2}$").sum()
        yyyy_mm = str_vals.str.match(r"^\d{4}-\d{1,2}$").sum()  # excludes YYYY-MM-DD

        lines.append("  Format pattern counts:")
        lines.append(f"    YYYY-MM-DD (full date): {yyyy_mm_dd:,}")
        lines.append(f"    YYYY-MM (month-year only): {yyyy_mm:,}")
        other_formats = len(s) - yyyy_mm_dd - yyyy_mm
        if other_formats > 0:
            lines.append(f"    Other formats: {other_formats:,}")
            other_vals = str_vals[~str_vals.str.match(r"^\d{4}-\d{1,2}(-\d{1,2})?$")]
            for v in other_vals.unique()[:5]:
                lines.append(f"      e.g. '{v}'")
        lines.append("")

        # Min/max if parseable
        parsed = pd.to_datetime(s, errors="coerce")
        valid = parsed.dropna()
        if len(valid) > 0:
            lines.append("  Parsed date range:")
            lines.append(f"    min: {valid.min()}")
            lines.append(f"    max: {valid.max()}")
        lines.append("")

    return "\n".join(lines)


def add_ongoing_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Add is_ongoing based on overall_status."""
    key = ongoing_status_key()
    ongoing_vals = set(key["ongoing"])
    df = df.copy()
    df["is_ongoing"] = df["overall_status"].fillna("").astype(str).isin(ongoing_vals)
    return df


def create_visualizations(df: pd.DataFrame) -> None:
    """Generate plots."""
    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = (10, 6)

    # 1. Phase distribution
    fig, ax = plt.subplots(figsize=(12, 6))
    phase_counts = df["phase"].value_counts()
    phase_counts.plot(kind="barh", ax=ax, color="steelblue", edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Count")
    ax.set_ylabel("Phase")
    ax.set_title("Study Phase Distribution")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "studies_phase_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved studies_phase_distribution.png")

    # 2. Overall status (ongoing vs completed)
    fig, ax = plt.subplots(figsize=(12, 6))
    status_counts = df["overall_status"].value_counts()
    status_counts.plot(kind="barh", ax=ax, color="coral", edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Count")
    ax.set_ylabel("Overall Status")
    ax.set_title("Study Overall Status (Key for Identifying Ongoing Studies)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "studies_overall_status.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved studies_overall_status.png")

    # 3. Ongoing vs completed pie
    df_with_flag = add_ongoing_flag(df)
    ongoing_counts = df_with_flag["is_ongoing"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(
        ongoing_counts.values,
        labels=["Completed/Other", "Ongoing"],
        autopct="%1.1f%%",
        colors=["#ff9999", "#66b3ff"],
        startangle=90,
    )
    ax.set_title("Ongoing vs Completed/Other Studies")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "studies_ongoing_pie.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved studies_ongoing_pie.png")

    # 4. Null counts for date columns (top 20)
    date_cols_flat = [c for cols in DATE_COLUMNS.values() for c in cols if c in df.columns]
    nulls = df[date_cols_flat].isna().sum().sort_values(ascending=True)
    nulls = nulls[nulls > 0].tail(25)  # top 25 by null count
    if len(nulls) > 0:
        fig, ax = plt.subplots(figsize=(10, 8))
        nulls.plot(kind="barh", ax=ax, color="teal", alpha=0.7)
        ax.set_xlabel("Null Count")
        ax.set_title("Null Counts for Date-Related Columns (Top 25)")
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "studies_date_nulls.png", dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Saved studies_date_nulls.png")


def print_report(df: pd.DataFrame) -> None:
    """Print text report to console and file."""
    lines = []
    lines.append("=" * 80)
    lines.append("STUDIES DATA EXPLORATION REPORT")
    lines.append("=" * 80)
    lines.append(f"Total rows: {len(df):,}")
    lines.append("")

    # Phase
    lines.append("-" * 40)
    lines.append("PHASE DISTRIBUTION")
    lines.append("-" * 40)
    phase_df = analyze_phase(df)
    for idx, row in phase_df.iterrows():
        lines.append(f"  {idx}: {int(row['count']):,} ({row['pct']}%)")
    lines.append("")

    # Date columns: start vs primary completion
    lines.append("-" * 40)
    lines.append("DATE COLUMNS: START vs PRIMARY COMPLETION vs COMPLETION")
    lines.append("-" * 40)
    lines.append("START dates: start_month_year, start_date_type, start_date")
    lines.append("PRIMARY COMPLETION: primary_completion_month_year, primary_completion_date_type, primary_completion_date")
    lines.append("COMPLETION: completion_month_year, completion_date_type, completion_date")
    lines.append("")
    date_analysis = analyze_date_columns(df)
    for col, info in sorted(date_analysis.items(), key=lambda x: (x[1]["group"], x[0])):
        grp = info["group"]
        nc = info["null_count"]
        pct = info["null_pct"]
        extra = ""
        if "min_date" in info:
            extra = f"  [range: {info['min_date'][:10]} to {info['max_date'][:10]}]"
        lines.append(f"  {col} ({grp}): {nc:,} null ({pct}%){extra}")
    lines.append("")

    # Overall status (ongoing)
    lines.append("-" * 40)
    lines.append("OVERALL STATUS (identifies ongoing vs completed)")
    lines.append("-" * 40)
    status_results = analyze_status(df)
    for col, counts in status_results.items():
        lines.append(f"  {col}:")
        for val, cnt in counts.head(15).items():
            lines.append(f"    {val}: {cnt:,}")
        lines.append("")
    lines.append("  Ongoing statuses: RECRUITING, ACTIVE_NOT_RECRUITING, NOT_YET_RECRUITING, etc.")
    lines.append("  Completed: COMPLETED, TERMINATED")
    lines.append("")

    # Null summary
    lines.append("-" * 40)
    lines.append("NULL COUNTS (top 20 columns)")
    lines.append("-" * 40)
    null_df = analyze_null_counts(df)
    for col, row in null_df.head(20).iterrows():
        lines.append(f"  {col}: {int(row['null_count']):,} ({row['null_pct']}%)")
    lines.append("")

    report = "\n".join(lines)
    print(report)
    (OUTPUT_DIR / "studies_report.txt").write_text(report)
    logger.info("Saved studies_report.txt")


def main() -> None:
    df = load_studies()
    print_report(df)
    create_visualizations(df)

    # Date format analysis (separate file)
    date_report = analyze_date_formats(df)
    date_report_path = OUTPUT_DIR / "studies_date_formats.txt"
    date_report_path.write_text(date_report)
    print(date_report)
    logger.info("Saved %s", date_report_path)

    logger.info("Done. Outputs in %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
