"""
Explore arm and intervention complexity: design_groups, interventions, browse_interventions.
Derived: number_of_interventions, intervention_type_mix, mono_therapy, has_placebo, etc.
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
    lines = []
    lines.append("=" * 70)
    lines.append("ARM & INTERVENTION COMPLEXITY EXPLORATION")
    lines.append("=" * 70)

    # design_groups
    dg_path = RAW_DATA / "design_groups.csv"
    if dg_path.exists():
        dg = pd.read_csv(dg_path, low_memory=False, nrows=100000)
        lines.append("")
        lines.append("-" * 50)
        lines.append("DESIGN_GROUPS (group_type)")
        lines.append("-" * 50)
        if "group_type" in dg.columns:
            lines.append("  Value counts:")
            for val, cnt in dg["group_type"].value_counts().head(15).items():
                lines.append(f"    {val}: {cnt:,}")
        lines.append("")
    else:
        lines.append("  design_groups.csv not found")
        lines.append("")

    # interventions (aggregate)
    int_path = RAW_DATA / "interventions.csv"
    if int_path.exists():
        interventions = pd.read_csv(int_path, low_memory=False, nrows=200000)
        n_int = interventions.groupby("nct_id").size()
        n_types = interventions.groupby("nct_id")["intervention_type"].nunique()
        lines.append("-" * 50)
        lines.append("INTERVENTIONS (per trial)")
        lines.append("-" * 50)
        lines.append(f"  Interventions per trial: min={n_int.min()}, max={n_int.max()}, median={n_int.median():.0f}")
        lines.append(f"  Unique intervention types per trial: min={n_types.min()}, max={n_types.max()}, median={n_types.median():.0f}")
        lines.append("")
    else:
        lines.append("  interventions.csv not found")
        lines.append("")

    # browse_interventions
    bi_path = RAW_DATA / "browse_interventions.csv"
    if bi_path.exists():
        bi = pd.read_csv(bi_path, low_memory=False, nrows=100000)
        lines.append("-" * 50)
        lines.append("BROWSE_INTERVENTIONS (MeSH terms)")
        lines.append("-" * 50)
        lines.append(f"  Columns: {list(bi.columns)}")
        if "downcase_mesh_term" in bi.columns:
            n_mesh = bi.groupby("nct_id")["downcase_mesh_term"].nunique()
            lines.append(f"  Unique MeSH terms per trial: min={n_mesh.min()}, max={n_mesh.max()}, median={n_mesh.median():.0f}")
        lines.append("")
    else:
        lines.append("  browse_interventions.csv not found")
        lines.append("")

    report = "\n".join(lines)
    print(report)
    (OUTPUT_DIR / "arm_intervention_report.txt").write_text(report)
    logger.info("Saved arm_intervention_report.txt")


if __name__ == "__main__":
    main()
