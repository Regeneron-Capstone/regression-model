"""
Explore site footprint: calculated_values, facilities, countries.
Derived features: number_of_facilities, number_of_countries, US-only, number_of_us_states,
geographic_dispersion, single-site, facility_density.
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
    lines.append("SITE FOOTPRINT EXPLORATION")
    lines.append("=" * 70)

    # calculated_values
    cv_path = RAW_DATA / "calculated_values.csv"
    if cv_path.exists():
        cv = pd.read_csv(cv_path, low_memory=False, nrows=100000)
        lines.append("")
        lines.append("-" * 50)
        lines.append("CALCULATED_VALUES (site-related columns)")
        lines.append("-" * 50)
        for col in ["number_of_facilities", "has_us_facility", "has_single_facility"]:
            if col in cv.columns:
                nulls = cv[col].isna().sum()
                lines.append(f"  {col}: nulls={nulls} ({nulls/len(cv)*100:.1f}%)")
                if col in ("has_us_facility", "has_single_facility"):
                    lines.append(f"    value_counts: {cv[col].value_counts().to_dict()}")
                else:
                    lines.append(f"    min={cv[col].min()}, max={cv[col].max()}, median={cv[col].median():.0f}")
        lines.append("")
    else:
        lines.append("  calculated_values.csv not found")
        lines.append("")

    # facilities
    fac_path = RAW_DATA / "facilities.csv"
    if fac_path.exists():
        fac = pd.read_csv(fac_path, low_memory=False, nrows=200000)
        lines.append("-" * 50)
        lines.append("FACILITIES")
        lines.append("-" * 50)
        lines.append(f"  Columns: {list(fac.columns)}")
        if "country" in fac.columns:
            us_count = (fac["country"].str.upper() == "UNITED STATES").sum()
            lines.append(f"  US facilities (sample): {us_count:,} ({us_count/len(fac)*100:.1f}%)")
        if "state" in fac.columns and "country" in fac.columns:
            us_fac = fac[fac["country"].str.upper() == "UNITED STATES"]
            lines.append(f"  Unique US states (sample): {us_fac['state'].nunique()}")
        if "nct_id" in fac.columns:
            fac_per_trial = fac.groupby("nct_id").size()
            lines.append(f"  Facilities per trial: min={fac_per_trial.min()}, max={fac_per_trial.max()}, median={fac_per_trial.median():.0f}")
        lines.append("")
    else:
        lines.append("  facilities.csv not found")
        lines.append("")

    # countries
    countries_path = RAW_DATA / "countries.csv"
    if countries_path.exists():
        countries = pd.read_csv(countries_path, low_memory=False, nrows=200000)
        lines.append("-" * 50)
        lines.append("COUNTRIES")
        lines.append("-" * 50)
        lines.append(f"  Columns: {list(countries.columns)}")
        if "nct_id" in countries.columns:
            cnt_per_trial = countries.groupby("nct_id").size()
            lines.append(f"  Countries per trial: min={cnt_per_trial.min()}, max={cnt_per_trial.max()}, median={cnt_per_trial.median():.0f}")
        if "name" in countries.columns:
            lines.append(f"  Top 10 countries: {list(countries['name'].value_counts().head(10).index)}")
        lines.append("")

    report = "\n".join(lines)
    print(report)
    (OUTPUT_DIR / "site_footprint_report.txt").write_text(report)
    logger.info("Saved site_footprint_report.txt")


if __name__ == "__main__":
    main()
