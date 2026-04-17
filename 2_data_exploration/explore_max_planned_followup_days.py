"""
Per-trial summary for max_planned_followup_days (max parsed time_frame in days
per NCT ID), aligned with 4_regression/core/step03_train_regression.py.

Writes a text report and optional CSV under 2_data_exploration/outputs/.
"""
import re
import logging
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
CLEAN_DATA = PROJECT_ROOT / "clean_data"
RAW_DATA = PROJECT_ROOT / "raw_data"
OUTPUT_DIR = PROJECT_ROOT / "2_data_exploration" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def parse_time_frame_days(tf: str) -> float | None:
    """Same logic as step03_train_regression parsing logic."""
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


def load_completed_trials() -> pd.DataFrame:
    path = CLEAN_DATA / "studies.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run 3_preprocessing/preprocess.py first.")
    studies = pd.read_csv(path, low_memory=False)
    studies = studies[studies["overall_status"] == "COMPLETED"].copy()
    return studies


def load_design_outcomes_for_nct_ids(nct_ids: set[str]) -> pd.DataFrame:
    do_path = RAW_DATA / "design_outcomes.csv"
    if not do_path.exists():
        raise FileNotFoundError(f"Missing {do_path}; run download_design_outcomes.")

    usecols = ["nct_id", "time_frame"]
    header = pd.read_csv(do_path, nrows=0, low_memory=False)
    if "time_frame" not in header.columns:
        raise ValueError("design_outcomes.csv has no time_frame column")
    chunks = []
    for chunk in pd.read_csv(do_path, chunksize=200_000, low_memory=False, usecols=usecols):
        sub = chunk[chunk["nct_id"].isin(nct_ids)]
        if len(sub):
            chunks.append(sub)
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=usecols)


def summarize_series(s: pd.Series, name: str, lines: list[str]) -> None:
    lines.append(f"  {name}")
    lines.append(f"    n (rows):        {len(s):,}")
    lines.append(f"    non-null:        {s.notna().sum():,}")
    lines.append(f"    null:            {s.isna().sum():,}  ({100 * s.isna().mean():.2f}%)")
    if s.notna().any():
        lines.append(f"    min:             {s.min():.2f}")
        lines.append(f"    max:             {s.max():.2f}")
        lines.append(f"    mean:            {s.mean():.2f}")
        lines.append(f"    std:             {s.std():.2f}")
        for p in (5, 25, 50, 75, 95):
            lines.append(f"    p{p}:              {s.quantile(p / 100):.2f}")


def main() -> None:
    studies = load_completed_trials()
    nct_ids = set(studies["nct_id"].astype(str))
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("max_planned_followup_days — per-trial study (COMPLETED cohort, 0_data/clean_data)")
    lines.append("=" * 72)
    lines.append(f"Completed trials in 0_data/clean_data/studies.csv: {len(studies):,}")
    lines.append("")

    do = load_design_outcomes_for_nct_ids(nct_ids)
    lines.append(f"design_outcomes rows for those NCT IDs: {len(do):,}")
    if len(do) == 0:
        lines.append("No matching rows; cannot compute max_planned_followup_days.")
        report = "\n".join(lines)
        print(report)
        (OUTPUT_DIR / "max_planned_followup_days_report.txt").write_text(report)
        return

    do["_tf_days"] = do["time_frame"].apply(parse_time_frame_days)
    n_parseable = do["_tf_days"].notna().sum()
    lines.append(
        f"Outcome rows with parseable time_frame: {n_parseable:,} "
        f"({100 * n_parseable / len(do):.2f}% of outcome rows)"
    )
    lines.append("")

    max_per = do.groupby("nct_id", as_index=False)["_tf_days"].max()
    max_per = max_per.rename(columns={"_tf_days": "max_planned_followup_days"})

    trial_df = studies[["nct_id", "phase"]].drop_duplicates(subset=["nct_id"])
    trial_df = trial_df.merge(max_per, on="nct_id", how="left")
    col = trial_df["max_planned_followup_days"]

    lines.append("-" * 72)
    lines.append("PER TRIAL (one row per nct_id in COMPLETED cohort)")
    lines.append("-" * 72)
    summarize_series(col, "max_planned_followup_days", lines)
    n_with_outcomes = do["nct_id"].nunique()
    n_missing_feature = col.isna().sum()
    lines.append("")
    lines.append(f"  Trials with ≥1 design_outcomes row:     {n_with_outcomes:,}")
    lines.append(
        f"  Trials with null max_planned_followup_days: {n_missing_feature:,} "
        f"({100 * n_missing_feature / len(trial_df):.2f}%)"
    )
    lines.append(
        "    (includes trials with no outcome rows, or no parseable time_frame on any row)"
    )
    lines.append("")

    plausible = col.dropna()
    for label, cap in (("> 10 years (3650 d)", 3650), ("> 20 years (7300 d)", 7300)):
        n = (plausible > cap).sum()
        lines.append(f"  Trials with max_planned_followup_days {label}: {n:,}")
    lines.append("")

    lines.append("-" * 72)
    lines.append("BY PHASE (max_planned_followup_days on COMPLETED trials)")
    lines.append("-" * 72)
    for phase in sorted(trial_df["phase"].dropna().unique()):
        sub = trial_df.loc[trial_df["phase"] == phase, "max_planned_followup_days"]
        lines.append(f"  {phase}  (n={len(sub):,})")
        lines.append(f"    non-null: {sub.notna().sum():,}  null: {sub.isna().sum():,}")
        if sub.notna().any():
            lines.append(
                f"    min={sub.min():.2f}  max={sub.max():.2f}  "
                f"median={sub.median():.2f}  mean={sub.mean():.2f}"
            )
    lines.append("")

    csv_path = OUTPUT_DIR / "max_planned_followup_days_per_trial.csv"
    trial_df.sort_values("nct_id").to_csv(csv_path, index=False)
    logger.info("Wrote %s", csv_path)

    report = "\n".join(lines)
    print(report)
    out_txt = OUTPUT_DIR / "max_planned_followup_days_report.txt"
    out_txt.write_text(report)
    logger.info("Wrote %s", out_txt)


if __name__ == "__main__":
    main()
