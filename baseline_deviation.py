"""
Prediction deviation analysis for per-phase trial duration models.

For each trial in the test set, computes how much the actual duration deviated
from the model's prediction, expressed as a percentage:

    pct_deviation = (actual_days - predicted_days) / (predicted_days + ε) × 100

Positive = trial ran longer than predicted (late).
Negative = trial finished earlier than predicted.

Outputs:
  results/predictions_with_deviation.csv  — per-trial predictions + deviation metrics
  results/deviation_summary.txt           — per-phase MAPE, accuracy bands, top-10 categories

Usage:
    python 5_analysis/baseline_deviation.py
    python 5_analysis/baseline_deviation.py --threshold 15
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split

# --- Path setup: import helpers from 4_regression without re-running its main() ---
PROJECT_ROOT = Path(__file__).parent.parent
_REGRESSION_DIR = PROJECT_ROOT / "4_regression"
if str(_REGRESSION_DIR) not in sys.path:
    sys.path.insert(0, str(_REGRESSION_DIR))

from train_regression import (  # noqa: E402
    load_and_join,
    prepare_features,
    KEPT_ELIGIBILITY,
    KEPT_ELIGIBILITY_CRITERIA_TEXT,
    KEPT_SITE_FOOTPRINT,
    KEPT_DESIGN,
    KEPT_ARM_INTERVENTION,
    KEPT_DESIGN_OUTCOMES,
    PHASES_WITH_DEDICATED_MODELS,
    TARGET_COLUMN,
)

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# --- Configurable threshold ---
LATE_THRESHOLD_PCT: float = 20.0  # flag trials whose actual duration exceeds prediction by this %


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def calculate_pct_deviation(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """
    Percentage deviation of actual from predicted duration.
    Epsilon prevents division by zero (predicted_days should never be 0 in practice).

    Uncomment the clip line below if extreme outliers (+/-5000%) distort reporting.
    """
    epsilon = 1e-10
    result = ((actual - predicted) / (predicted + epsilon)) * 100
    # result = np.clip(result, -200, 200)
    return result


def is_late(pct_deviation: float, threshold: float = LATE_THRESHOLD_PCT) -> bool:
    """True if actual duration exceeded prediction by more than threshold %."""
    return pct_deviation > threshold


# ---------------------------------------------------------------------------
# Model training (mirrors train_regression.py loop exactly)
# ---------------------------------------------------------------------------

def train_phase_models(df: pd.DataFrame, random_state: int = 42) -> dict[str, dict]:
    """
    Train one HistGradientBoostingRegressor per phase, identical to train_regression.py.
    Returns a dict keyed by phase label with model and test-set data.
    """
    prep_kw = dict(
        eligibility_columns=KEPT_ELIGIBILITY,
        eligibility_criteria_text_columns=KEPT_ELIGIBILITY_CRITERIA_TEXT,
        site_footprint_columns=KEPT_SITE_FOOTPRINT,
        design_columns=KEPT_DESIGN,
        arm_intervention_columns=KEPT_ARM_INTERVENTION,
        design_outcomes_columns=KEPT_DESIGN_OUTCOMES,
        encode_phase=False,
    )

    results: dict[str, dict] = {}

    for phase in PHASES_WITH_DEDICATED_MODELS:
        df_p = df[df["phase"].astype(str) == phase].copy()

        if len(df_p) < 30:
            logger.info("Skipping %s — too few rows (%d)", phase, len(df_p))
            continue

        # Keep metadata-aligned copy BEFORE prepare_features drops NaN target rows
        df_p_valid = df_p.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)

        X, y, _, _ = prepare_features(df_p, **prep_kw)

        if len(y) < 30:
            logger.info("Skipping %s — too few rows after dropna (%d)", phase, len(y))
            continue

        # Identical two-step 60/20/20 split from train_regression.py
        X_train, X_temp, y_train, y_temp, idx_train, idx_temp = train_test_split(
            X, y, np.arange(len(y)), test_size=0.4, random_state=random_state
        )
        X_val, X_test, y_val, y_test, idx_val, idx_test = train_test_split(
            X_temp, y_temp, idx_temp, test_size=0.5, random_state=random_state
        )

        model = TransformedTargetRegressor(
            regressor=HistGradientBoostingRegressor(max_iter=200, random_state=random_state),
            func=np.log1p,
            inverse_func=np.expm1,
        )
        model.fit(X_train, y_train)
        logger.info(
            "Trained %s  train=%d  val=%d  test=%d",
            phase, len(y_train), len(y_val), len(y_test),
        )

        results[phase] = {
            "model": model,
            "X_test": X_test,
            "y_test": y_test,
            "df_test_rows": df_p_valid.iloc[idx_test].reset_index(drop=True),
        }

    return results


# ---------------------------------------------------------------------------
# Deviation table
# ---------------------------------------------------------------------------

def generate_deviation_table(
    phase_results: dict[str, dict],
    threshold: float = LATE_THRESHOLD_PCT,
) -> pd.DataFrame:
    """
    For each phase model, predict on the test set and compute deviation metrics.
    Returns a single DataFrame across all phases.
    """
    frames: list[pd.DataFrame] = []

    for phase, data in phase_results.items():
        model = data["model"]
        X_test = data["X_test"]
        y_test = data["y_test"]
        meta = data["df_test_rows"]

        predicted_days = model.predict(X_test)
        actual_days = y_test

        pct_dev = calculate_pct_deviation(actual_days, predicted_days)
        abs_error = np.abs(actual_days - predicted_days)
        late = np.array([is_late(v, threshold) for v in pct_dev])

        frame = pd.DataFrame({
            "nct_id": meta["nct_id"].values if "nct_id" in meta.columns else np.full(len(y_test), ""),
            "phase": phase,
            "category": meta["category"].values if "category" in meta.columns else np.full(len(y_test), ""),
            "actual_days": actual_days,
            "predicted_days": predicted_days,
            "pct_deviation": pct_dev,
            "abs_error_days": abs_error,
            "late_flag": late,
        })
        frames.append(frame)

    if not frames:
        logger.warning("No phase models produced results.")
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def print_summary(
    df: pd.DataFrame,
    threshold: float = LATE_THRESHOLD_PCT,
    output_path: Path | None = None,
) -> None:
    """Print per-phase deviation stats, accuracy bands, and top-10 category breakdown."""
    lines: list[str] = []

    lines.append("=" * 70)
    lines.append("PREDICTION DEVIATION ANALYSIS")
    lines.append(f"Late threshold: >{threshold:.0f}% over prediction")
    lines.append(f"Total test trials: {len(df):,}")
    lines.append("=" * 70)

    for phase in PHASES_WITH_DEDICATED_MODELS:
        phase_df = df[df["phase"] == phase]
        if phase_df.empty:
            continue

        pct_dev = phase_df["pct_deviation"]
        mape = np.mean(np.abs(pct_dev))
        mae_days = phase_df["abs_error_days"].mean()
        n_late = phase_df["late_flag"].sum()

        within_10 = (np.abs(pct_dev) <= 10).mean() * 100
        within_20 = (np.abs(pct_dev) <= 20).mean() * 100
        within_30 = (np.abs(pct_dev) <= 30).mean() * 100

        lines.append("")
        lines.append(f"PHASE: {phase}")
        lines.append("=" * 70)
        lines.append(f"  Sample size:              {len(phase_df):,} trials")
        lines.append(f"  MAE (days):               {mae_days:.1f}")
        lines.append(f"  MAPE (%):                 {mape:.1f}%")
        lines.append(f"  Mean % deviation:         {pct_dev.mean():.1f}%")
        lines.append(f"  Median % deviation:       {pct_dev.median():.1f}%")
        lines.append(f"  Std deviation:            {pct_dev.std():.1f}%")
        lines.append(f"  P25:                      {pct_dev.quantile(0.25):.1f}%")
        lines.append(f"  P75:                      {pct_dev.quantile(0.75):.1f}%")
        lines.append(f"  P90:                      {pct_dev.quantile(0.90):.1f}%")
        lines.append(f"  Trials flagged as late:   {n_late:,} ({phase_df['late_flag'].mean() * 100:.1f}%)")
        lines.append("")
        lines.append("  Accuracy bands:")
        lines.append(f"    Within ±10%:            {within_10:.1f}%")
        lines.append(f"    Within ±20%:            {within_20:.1f}%")
        lines.append(f"    Within ±30%:            {within_30:.1f}%")
        lines.append("=" * 70)

    # Per-category breakdown (top 10 by trial count)
    if "category" in df.columns and df["category"].notna().any() and (df["category"] != "").any():
        lines.append("")
        lines.append("PERFORMANCE BY DISEASE CATEGORY (Top 10 by trial count)")
        lines.append("=" * 70)

        cat_stats = (
            df.groupby("category")
            .agg(
                N_trials=("pct_deviation", "count"),
                Mean_pct_dev=("pct_deviation", "mean"),
                MAPE=("pct_deviation", lambda x: np.mean(np.abs(x))),
                Late_rate=("late_flag", "mean"),
            )
            .sort_values("N_trials", ascending=False)
            .head(10)
            .round(2)
        )
        cat_stats["Late_rate"] = (cat_stats["Late_rate"] * 100).round(1)
        lines.append(cat_stats.to_string())
        lines.append("=" * 70)

    report = "\n".join(lines)
    print(report)

    save_path = output_path or RESULTS_DIR / "deviation_summary.txt"
    save_path.write_text(report)
    logger.info("Saved %s", save_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(threshold: float = LATE_THRESHOLD_PCT) -> None:
    logger.info("Loading data...")
    df = load_and_join(
        eligibility_columns=KEPT_ELIGIBILITY,
        site_footprint_columns=KEPT_SITE_FOOTPRINT,
        design_columns=KEPT_DESIGN,
        arm_intervention_columns=KEPT_ARM_INTERVENTION,
        design_outcomes_columns=KEPT_DESIGN_OUTCOMES,
    )

    logger.info("Training per-phase models...")
    phase_results = train_phase_models(df)

    logger.info("Generating deviation table...")
    deviation_df = generate_deviation_table(phase_results, threshold=threshold)

    out_csv = RESULTS_DIR / "predictions_with_deviation.csv"
    deviation_df.to_csv(out_csv, index=False)
    logger.info("Saved %s", out_csv)

    print_summary(
        deviation_df,
        threshold=threshold,
        output_path=RESULTS_DIR / "deviation_summary.txt",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trial duration prediction deviation analysis")
    parser.add_argument(
        "--threshold",
        type=float,
        default=LATE_THRESHOLD_PCT,
        help="Percent deviation above prediction to flag as late (default: %(default)s)",
    )
    args = parser.parse_args()
    main(threshold=args.threshold)
