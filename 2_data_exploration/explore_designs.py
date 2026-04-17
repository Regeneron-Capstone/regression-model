"""
Explore designs: allocation, intervention_model, primary_purpose, masking, role flags.
Derived: randomized, intervention_model type, masking_depth_score, primary_purpose category.
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
    path = RAW_DATA / "designs.csv"
    if not path.exists():
        logger.error("designs.csv not found. Run: python 1_scripts/download_designs.py")
        return

    df = pd.read_csv(path, low_memory=False)

    lines = []
    lines.append("=" * 70)
    lines.append("DESIGNS EXPLORATION")
    lines.append("=" * 70)
    lines.append(f"Rows: {len(df):,}")
    lines.append(f"Columns: {list(df.columns)}")
    lines.append("")

    for col in ["allocation", "intervention_model", "primary_purpose", "masking"]:
        if col in df.columns:
            lines.append("-" * 50)
            lines.append(col.upper())
            lines.append("-" * 50)
            nulls = df[col].isna().sum()
            lines.append(f"  Nulls: {nulls:,} ({nulls/len(df)*100:.1f}%)")
            lines.append("  Value counts:")
            for val, cnt in df[col].value_counts().head(15).items():
                lines.append(f"    {val}: {cnt:,}")
            lines.append("")

    for col in ["subject_masked", "caregiver_masked", "investigator_masked", "outcomes_assessor_masked"]:
        if col in df.columns:
            lines.append("-" * 50)
            lines.append(col.upper())
            lines.append("-" * 50)
            lines.append(f"  {df[col].value_counts().to_dict()}")
            lines.append("")

    report = "\n".join(lines)
    print(report)
    (OUTPUT_DIR / "designs_report.txt").write_text(report)
    logger.info("Saved designs_report.txt")


if __name__ == "__main__":
    main()
