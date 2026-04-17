"""
Explore design_outcomes: outcome_type, measure, time_frame, population, description.
Derive features for duration prediction:
- max/median planned follow-up from time_frame
- number of primary/secondary outcomes
- survival/imaging/lab/PRO/safety endpoint flags
- endpoint-complexity score
- multiple assessment windows
"""
import re
import logging
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
RAW_DATA = PROJECT_ROOT / "raw_data"
OUTPUT_DIR = PROJECT_ROOT / "2_data_exploration" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def parse_time_frame_days(tf: str) -> float | None:
    """Parse time_frame string to days. Returns None if unparseable."""
    if pd.isna(tf) or not isinstance(tf, str) or not tf.strip():
        return None
    tf = tf.strip().lower()
    # Patterns: "12 months", "2 years", "52 weeks", "365 days", "1 year"
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


def has_endpoint_type(text: str, keywords: list[str]) -> bool:
    """Check if text contains any of the keywords (case-insensitive)."""
    if pd.isna(text) or not isinstance(text, str):
        return False
    t = text.lower()
    return any(k in t for k in keywords)


def main() -> None:
    path = RAW_DATA / "design_outcomes.csv"
    if not path.exists():
        logger.error("design_outcomes.csv not found. Run: python 1_scripts/download_design_outcomes.py")
        return

    df = pd.read_csv(path, low_memory=False)

    lines = []
    lines.append("=" * 70)
    lines.append("DESIGN_OUTCOMES EXPLORATION")
    lines.append("=" * 70)
    lines.append(f"Rows: {len(df):,}")
    lines.append(f"Unique NCT IDs: {df['nct_id'].nunique():,}")
    lines.append(f"Columns: {list(df.columns)}")
    lines.append("")

    # outcome_type
    if "outcome_type" in df.columns:
        lines.append("-" * 50)
        lines.append("OUTCOME_TYPE")
        lines.append("-" * 50)
        nulls = df["outcome_type"].isna().sum()
        lines.append(f"  Nulls: {nulls:,} ({nulls/len(df)*100:.1f}%)")
        lines.append("  Value counts:")
        for val, cnt in df["outcome_type"].value_counts().head(20).items():
            lines.append(f"    {val}: {cnt:,}")
        lines.append("")

    # measure
    if "measure" in df.columns:
        lines.append("-" * 50)
        lines.append("MEASURE (sample)")
        lines.append("-" * 50)
        lines.append(f"  Nulls: {df['measure'].isna().sum():,}")
        lines.append("  Sample values:")
        for v in df["measure"].dropna().head(5).tolist():
            lines.append(f"    {str(v)[:100]}...")
        lines.append("")

    # time_frame
    if "time_frame" in df.columns:
        lines.append("-" * 50)
        lines.append("TIME_FRAME")
        lines.append("-" * 50)
        nulls = df["time_frame"].isna().sum()
        lines.append(f"  Nulls: {nulls:,} ({nulls/len(df)*100:.1f}%)")
        # Parse
        df["_tf_days"] = df["time_frame"].apply(parse_time_frame_days)
        parsed = df["_tf_days"].notna().sum()
        lines.append(f"  Parseable: {parsed:,} ({parsed/len(df)*100:.1f}%)")
        lines.append("  Sample values:")
        for v in df["time_frame"].dropna().head(5).tolist():
            lines.append(f"    {str(v)[:80]}...")
        lines.append("")

    # population
    if "population" in df.columns:
        lines.append("-" * 50)
        lines.append("POPULATION")
        lines.append("-" * 50)
        nulls = df["population"].isna().sum()
        lines.append(f"  Nulls: {nulls:,} ({nulls/len(df)*100:.1f}%)")
        lines.append("  Value counts (top 10):")
        for val, cnt in df["population"].value_counts().head(10).items():
            lines.append(f"    {val}: {cnt:,}")
        lines.append("")

    # description
    if "description" in df.columns:
        lines.append("-" * 50)
        lines.append("DESCRIPTION (sample)")
        lines.append("-" * 50)
        lines.append(f"  Nulls: {df['description'].isna().sum():,}")
        lines.append("")

    # Per-trial aggregates
    lines.append("-" * 50)
    lines.append("PER-TRIAL AGGREGATES (sample)")
    lines.append("-" * 50)
    agg = df.groupby("nct_id").size().reset_index(name="n_outcomes")
    if "outcome_type" in df.columns:
        n_prim = df.groupby("nct_id")["outcome_type"].apply(lambda x: (x.fillna("").str.upper() == "PRIMARY").sum()).reset_index(name="n_primary")
        n_sec = df.groupby("nct_id")["outcome_type"].apply(lambda x: (x.fillna("").str.upper() == "SECONDARY").sum()).reset_index(name="n_secondary")
        agg = agg.merge(n_prim, on="nct_id", how="left").merge(n_sec, on="nct_id", how="left")
    max_tf = df.groupby("nct_id")["_tf_days"].max().reset_index(name="max_tf_days")
    med_tf = df.groupby("nct_id")["_tf_days"].median().reset_index(name="median_tf_days")
    agg = agg.merge(max_tf, on="nct_id", how="left").merge(med_tf, on="nct_id", how="left")
    lines.append(f"  Trials with outcomes: {len(agg):,}")
    lines.append(f"  n_outcomes: min={agg['n_outcomes'].min()}, max={agg['n_outcomes'].max()}, median={agg['n_outcomes'].median():.0f}")
    if "max_tf_days" in agg.columns:
        lines.append(f"  max_tf_days: non-null={agg['max_tf_days'].notna().sum():,}, median={agg['max_tf_days'].median():.0f}")
    lines.append("")

    # Endpoint type flags (measure + description)
    SURVIVAL_KEYWORDS = ["survival", "os", "pfs", "dfs", "overall survival", "progression-free survival"]
    IMAGING_KEYWORDS = ["imaging", "mri", "ct scan", "pet", "radiograph", "ultrasound"]
    LAB_KEYWORDS = ["laboratory", "lab", "blood", "serum", "plasma", "biomarker", "pk", "pd"]
    PRO_KEYWORDS = ["quality of life", "qol", "patient-reported", "pro", "questionnaire", "scale", "score"]
    SAFETY_KEYWORDS = ["safety", "adverse", "ae", "sae", "toxicity", "tolerability"]

    def check_text(text: str, keywords: list[str]) -> bool:
        return has_endpoint_type(text, keywords)

    for col in ["measure", "description"]:
        if col in df.columns:
            df[f"_has_survival_{col}"] = df[col].apply(lambda x: check_text(x, SURVIVAL_KEYWORDS))
            df[f"_has_imaging_{col}"] = df[col].apply(lambda x: check_text(x, IMAGING_KEYWORDS))
            df[f"_has_lab_{col}"] = df[col].apply(lambda x: check_text(x, LAB_KEYWORDS))
            df[f"_has_pro_{col}"] = df[col].apply(lambda x: check_text(x, PRO_KEYWORDS))
            df[f"_has_safety_{col}"] = df[col].apply(lambda x: check_text(x, SAFETY_KEYWORDS))

    def or_cols(a: str, b: str) -> pd.Series:
        sa = df[a] if a in df.columns else pd.Series(False, index=df.index)
        sb = df[b] if b in df.columns else pd.Series(False, index=df.index)
        return sa.fillna(False) | sb.fillna(False)

    df["_has_survival"] = or_cols("_has_survival_measure", "_has_survival_description")
    df["_has_imaging"] = or_cols("_has_imaging_measure", "_has_imaging_description")
    df["_has_lab"] = or_cols("_has_lab_measure", "_has_lab_description")
    df["_has_pro"] = or_cols("_has_pro_measure", "_has_pro_description")
    df["_has_safety"] = or_cols("_has_safety_measure", "_has_safety_description")

    flag_cols = ["_has_survival", "_has_imaging", "_has_lab", "_has_pro", "_has_safety"]
    flag_cols = [c for c in flag_cols if c in df.columns]
    if flag_cols:
        lines.append("-" * 50)
        lines.append("ENDPOINT TYPE FLAGS (per outcome)")
        lines.append("-" * 50)
        for c in flag_cols:
            pct = df[c].sum() / len(df) * 100
            lines.append(f"  {c.replace('_has_', '')}: {df[c].sum():,} ({pct:.1f}%)")
        lines.append("")

    report = "\n".join(lines)
    print(report)
    (OUTPUT_DIR / "design_outcomes_report.txt").write_text(report)
    logger.info("Saved design_outcomes_report.txt")


if __name__ == "__main__":
    main()
