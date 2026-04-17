"""
Data exploration for sponsors.csv: agency_class distribution.
"""
import logging
from pathlib import Path

import matplotlib.pyplot as plt
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


def load_sponsors() -> pd.DataFrame:
    path = RAW_DATA / "sponsors.csv"
    logger.info("Loading %s", path)
    df = pd.read_csv(path, low_memory=False)
    logger.info("Loaded %s rows", f"{len(df):,}")
    return df


def analyze_agency_class(df: pd.DataFrame) -> pd.DataFrame:
    """Agency class distribution: counts and percentages."""
    counts = df["agency_class"].value_counts(dropna=False)
    pct = (counts / len(df) * 100).round(1)
    return pd.DataFrame({"count": counts, "pct": pct}).sort_values("count", ascending=False)


def create_visualizations(df: pd.DataFrame) -> None:
    """Generate agency_class distribution plots."""
    sns.set_style("whitegrid")

    # 1. Horizontal bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    agency_df = analyze_agency_class(df)
    agency_df["count"].plot(
        kind="barh",
        ax=ax,
        color=sns.color_palette("husl", len(agency_df)),
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_xlabel("Count")
    ax.set_ylabel("Agency Class")
    ax.set_title("Sponsor Agency Class Distribution")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "sponsors_agency_class_bar.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved sponsors_agency_class_bar.png")

    # 2. Pie chart
    fig, ax = plt.subplots(figsize=(8, 8))
    agency_df = analyze_agency_class(df)
    colors = sns.color_palette("Set3", len(agency_df))
    wedges, texts, autotexts = ax.pie(
        agency_df["count"],
        labels=agency_df.index,
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
        pctdistance=0.75,
    )
    for autotext in autotexts:
        autotext.set_fontsize(9)
    ax.set_title("Sponsor Agency Class Distribution")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "sponsors_agency_class_pie.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved sponsors_agency_class_pie.png")

    # 3. Lead vs collaborator by agency class
    if "lead_or_collaborator" in df.columns:
        cross = pd.crosstab(df["agency_class"], df["lead_or_collaborator"])
        fig, ax = plt.subplots(figsize=(10, 6))
        cross.plot(kind="bar", stacked=True, ax=ax, colormap="Pastel1")
        ax.set_xlabel("Agency Class")
        ax.set_ylabel("Count")
        ax.set_title("Agency Class by Lead vs Collaborator")
        ax.legend(title="Role")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "sponsors_agency_by_role.png", dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Saved sponsors_agency_by_role.png")


def print_report(df: pd.DataFrame) -> None:
    """Print text report."""
    lines = []
    lines.append("=" * 80)
    lines.append("SPONSORS DATA EXPLORATION REPORT")
    lines.append("=" * 80)
    lines.append(f"Total rows: {len(df):,}")
    lines.append(f"Unique NCT IDs: {df['nct_id'].nunique():,}")
    lines.append("")
    lines.append("-" * 40)
    lines.append("AGENCY CLASS DISTRIBUTION")
    lines.append("-" * 40)
    agency_df = analyze_agency_class(df)
    for idx, row in agency_df.iterrows():
        lines.append(f"  {idx}: {int(row['count']):,} ({row['pct']}%)")
    lines.append("")

    report = "\n".join(lines)
    print(report)
    (OUTPUT_DIR / "sponsors_report.txt").write_text(report)
    logger.info("Saved sponsors_report.txt")


def main() -> None:
    df = load_sponsors()
    print_report(df)
    create_visualizations(df)
    logger.info("Done. Outputs in %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
