"""
Explore interventions: columns, intervention_type distribution, nulls.
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
    path = RAW_DATA / "interventions.csv"
    if not path.exists():
        logger.error("interventions.csv not found. Run: python 1_scripts/download_interventions.py")
        return

    df = pd.read_csv(path, low_memory=False, nrows=200000)

    lines = []
    lines.append("=" * 70)
    lines.append("INTERVENTIONS EXPLORATION")
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

    # intervention_type
    if "intervention_type" in df.columns:
        lines.append("-" * 50)
        lines.append("INTERVENTION_TYPE")
        lines.append("-" * 50)
        lines.append(f"  Unique values: {df['intervention_type'].nunique():,}")
        lines.append("  Value counts:")
        for val, cnt in df["intervention_type"].value_counts().items():
            lines.append(f"    {val}: {cnt:,}")
        lines.append("")
    else:
        lines.append("  intervention_type column not found")
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
    (OUTPUT_DIR / "interventions_report.txt").write_text(report)
    logger.info("Saved interventions_report.txt")


if __name__ == "__main__":
    main()
