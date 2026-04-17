"""
Explore browse_conditions: columns, downcase_mesh_term distribution, nulls.
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


def main() -> None:
    path = RAW_DATA / "browse_conditions.csv"
    if not path.exists():
        logger.error("browse_conditions.csv not found. Run: python 1_scripts/download_browse_conditions.py")
        return

    # Sample first 200k rows for speed (full table ~4M rows)
    df = pd.read_csv(path, low_memory=False, nrows=200000)

    lines = []
    lines.append("=" * 70)
    lines.append("BROWSE_CONDITIONS EXPLORATION")
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

    # downcase_mesh_term
    if "downcase_mesh_term" in df.columns:
        lines.append("-" * 50)
        lines.append("DOWNCASE_MESH_TERM")
        lines.append("-" * 50)
        lines.append(f"  Unique values: {df['downcase_mesh_term'].nunique():,}")
        lines.append("  Top 20 by count:")
        for term, cnt in df["downcase_mesh_term"].value_counts().head(20).items():
            lines.append(f"    {term}: {cnt:,}")
        lines.append("")
    elif "mesh_term" in df.columns:
        lines.append("-" * 50)
        lines.append("MESH_TERM (downcase_mesh_term not found)")
        lines.append("-" * 50)
        lines.append(f"  Unique values: {df['mesh_term'].nunique():,}")
        lines.append("  Top 20 by count:")
        for term, cnt in df["mesh_term"].value_counts().head(20).items():
            lines.append(f"    {term}: {cnt:,}")
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
    (OUTPUT_DIR / "browse_conditions_report.txt").write_text(report)
    logger.info("Saved browse_conditions_report.txt")


if __name__ == "__main__":
    main()
