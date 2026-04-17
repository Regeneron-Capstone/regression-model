"""
Explore eligibility *criteria* free text in eligibilities.csv for future NLP-style features:
text length, inclusion/exclusion tilde counts, procedure-burden keywords.

Data: raw_data/eligibilities.csv — column `criteria` holds the full text block.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent

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

RAW_DATA = PROJECT_ROOT / "raw_data"
OUTPUT_DIR = PROJECT_ROOT / "2_data_exploration" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


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


def summarize_series(s: pd.Series, name: str, lines: list[str]) -> None:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) == 0:
        lines.append(f"  {name}: no values")
        return
    lines.append(f"  {name}: n={len(s):,}  mean={s.mean():.2f}  std={s.std():.2f}")
    lines.append(
        f"    min={s.min():.0f}  p25={s.quantile(0.25):.0f}  "
        f"median={s.median():.0f}  p75={s.quantile(0.75):.0f}  max={s.max():.0f}"
    )


def main() -> None:
    path = RAW_DATA / "eligibilities.csv"
    if not path.exists():
        logger.error("eligibilities.csv not found. Run: python 1_scripts/download_eligibilities.py")
        return

    df = pd.read_csv(path, usecols=["nct_id", "criteria"], low_memory=False)
    n = len(df)

    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("ELIGIBILITY CRITERIA TEXT (for NLP-style features)")
    lines.append("=" * 72)
    lines.append(f"Source: {path}")
    lines.append(f"Rows: {n:,}  |  Unique nct_id: {df['nct_id'].nunique():,}")
    lines.append("")
    lines.append("All columns in eligibilities.csv (full file):")
    full_cols = pd.read_csv(path, nrows=0).columns.tolist()
    lines.append(f"  {full_cols}")
    lines.append("")
    lines.append("This report focuses on: nct_id, criteria")
    lines.append("")

    crit = df["criteria"]
    non_null = crit.notna()
    non_empty = non_null & crit.astype(str).str.strip().ne("") & crit.astype(str).ne("nan")
    lines.append("-" * 50)
    lines.append("CRITERIA COLUMN COVERAGE")
    lines.append("-" * 50)
    lines.append(f"  Non-null:     {non_null.sum():,} ({non_null.mean()*100:.2f}%)")
    lines.append(f"  Non-empty:    {non_empty.sum():,} ({non_empty.mean()*100:.2f}%)")
    lines.append("")

    text = crit.fillna("").astype(str)
    lengths = text.str.len()
    has_incl = text.str.contains("Inclusion Criteria:", regex=False, na=False)
    has_excl = text.str.contains("Exclusion Criteria:", regex=False, na=False)
    has_both = has_incl & has_excl

    lines.append("-" * 50)
    lines.append("HEADER / FORMAT FLAGS (ClinicalTrials.gov style)")
    lines.append("-" * 50)
    lines.append(f"  Contains 'Inclusion Criteria:':   {has_incl.sum():,} ({has_incl.mean()*100:.2f}%)")
    lines.append(f"  Contains 'Exclusion Criteria:':    {has_excl.sum():,} ({has_excl.mean()*100:.2f}%)")
    lines.append(f"  Contains both:                     {has_both.sum():,} ({has_both.mean()*100:.2f}%)")
    lines.append(f"  Contains '~' (tilde):              {text.str.contains('~', regex=False, na=False).sum():,}")
    lines.append("")

    inc_counts = text.map(count_inclusion_tildes)
    exc_counts = text.map(count_exclusion_tildes)
    burden = text.map(has_burden_keyword)

    lines.append("-" * 50)
    lines.append("DERIVED FEATURES (slide logic; all rows)")
    lines.append("-" * 50)
    summarize_series(lengths[non_empty], "criteria length (chars)", lines)
    lines.append("")
    summarize_series(inc_counts[non_empty], "inclusion ~ count", lines)
    lines.append("")
    summarize_series(exc_counts[non_empty], "exclusion ~ count", lines)
    lines.append("")
    lines.append(
        f"  has_burden_keyword: {burden.sum():,} true ({burden.mean()*100:.2f}%) "
        f"(keywords: {len(BURDEN_KEYWORDS)} terms, see script)"
    )
    lines.append("")

    lines.append("-" * 50)
    lines.append("EXAMPLE SNIPPETS (first 3 non-empty with both headers)")
    lines.append("-" * 50)
    show = df.loc[non_empty & has_both, "criteria"].head(3)
    for i, raw in enumerate(show, 1):
        s = str(raw)[:900]
        lines.append(f"  --- example {i} (first 900 chars) ---")
        lines.append("  " + s.replace("\n", " ") + ("..." if len(str(raw)) > 900 else ""))
        lines.append("")

    report = "\n".join(lines)
    print(report)
    out = OUTPUT_DIR / "eligibility_criteria_text_report.txt"
    out.write_text(report)
    logger.info("Saved %s", out)


if __name__ == "__main__":
    main()
