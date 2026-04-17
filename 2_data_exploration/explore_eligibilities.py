"""
Explore eligibilities: columns, gender, minimum_age, maximum_age, adult, child, older_adult.
"""
import logging
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
RAW_DATA = PROJECT_ROOT / "raw_data"
OUTPUT_DIR = PROJECT_ROOT / "2_data_exploration" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

ELIGIBILITY_COLS = ["gender", "minimum_age", "maximum_age", "adult", "child", "older_adult"]


def main() -> None:
    path = RAW_DATA / "eligibilities.csv"
    if not path.exists():
        logger.error("eligibilities.csv not found. Run: python 1_scripts/download_eligibilities.py")
        return

    df = pd.read_csv(path, low_memory=False, nrows=500000)

    lines = []
    lines.append("=" * 70)
    lines.append("ELIGIBILITIES EXPLORATION")
    lines.append("=" * 70)
    lines.append(f"Rows (sample): {len(df):,}")
    lines.append(f"Columns: {list(df.columns)}")
    lines.append("")

    # Null counts
    lines.append("-" * 50)
    lines.append("NULL COUNTS")
    lines.append("-" * 50)
    for col in df.columns:
        nulls = df[col].isna().sum()
        pct = nulls / len(df) * 100
        lines.append(f"  {col}: {nulls:,} ({pct:.1f}%)")
    lines.append("")

    # Target columns
    for col in ELIGIBILITY_COLS:
        if col in df.columns:
            lines.append("-" * 50)
            lines.append(col.upper())
            lines.append("-" * 50)
            nulls = df[col].isna().sum()
            lines.append(f"  Nulls: {nulls:,} ({nulls/len(df)*100:.1f}%)")
            if df[col].dtype in ["object", "bool"] or df[col].nunique() < 50:
                lines.append(f"  Unique values: {df[col].nunique():,}")
                for val, cnt in df[col].value_counts().head(15).items():
                    lines.append(f"    {val}: {cnt:,}")
            else:
                lines.append(f"  Min: {df[col].min()}, Max: {df[col].max()}, Median: {df[col].median()}")
            lines.append("")
        else:
            lines.append(f"  {col}: column not found")
            lines.append("")

    # Rows per nct_id
    if "nct_id" in df.columns:
        rows_per_trial = df.groupby("nct_id").size()
        lines.append("-" * 50)
        lines.append("ROWS PER TRIAL (nct_id)")
        lines.append("-" * 50)
        lines.append(f"  Unique nct_ids: {df['nct_id'].nunique():,}")
        lines.append(f"  Min rows/trial: {rows_per_trial.min()}")
        lines.append(f"  Max rows/trial: {rows_per_trial.max()}")
        lines.append(f"  Median rows/trial: {rows_per_trial.median():.0f}")
        lines.append("")

    report = "\n".join(lines)
    print(report)
    (OUTPUT_DIR / "eligibilities_report.txt").write_text(report)
    logger.info("Saved eligibilities_report.txt")


if __name__ == "__main__":
    main()
