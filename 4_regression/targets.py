"""
Target construction and deviation metrics for trial duration modeling.

Formulas match ClinicalTrials.gov-style date fields on `studies`:
- start_date
- primary_completion_date  (end of primary data collection / primary completion)
- completion_date         (study completion)

All day deltas use calendar-day difference via pandas (same as preprocess.py for duration_days).
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

# Column name materialized on clean studies by preprocess.py; regression reads this from CSV.
TARGET_DURATION_COLUMN: str = "duration_days"

# Internal column name used during feature assembly (not persisted to CSV).
MODEL_TARGET_INTERNAL_COL: str = "__model_target_days__"

TargetKind = Literal["primary_completion", "post_primary_completion", "total_completion"]

DEFAULT_TARGET_KIND: TargetKind = "primary_completion"

# Epsilon for division in percent deviation (matches 5_deviation historical behavior).
_PCT_DEV_EPSILON: float = 1e-10


def compute_days_to_primary_completion(df: pd.DataFrame) -> pd.Series:
    """
    Primary-phase span in whole days:

        primary_completion_date - start_date

    Parsed with ``errors="coerce"``. Result is float with NaN where either date is missing/invalid.
    Matches ``preprocess.py`` assignment to ``duration_days`` before filtering.
    """
    start = pd.to_datetime(df["start_date"], errors="coerce")
    primary_end = pd.to_datetime(df["primary_completion_date"], errors="coerce")
    return (primary_end - start).dt.days.astype(float)


def compute_days_post_primary_completion(df: pd.DataFrame) -> pd.Series:
    """
    Post-primary window in whole days (follow-up / wind-down after primary completion):

        completion_date - primary_completion_date

    NaN if either side is missing/invalid.
    """
    primary_end = pd.to_datetime(df["primary_completion_date"], errors="coerce")
    completion = pd.to_datetime(df["completion_date"], errors="coerce")
    return (completion - primary_end).dt.days.astype(float)


def compute_days_total_completion(df: pd.DataFrame) -> pd.Series:
    """
    Full study span in whole days:

        completion_date - start_date

    NaN if either side is missing/invalid.
    """
    start = pd.to_datetime(df["start_date"], errors="coerce")
    completion = pd.to_datetime(df["completion_date"], errors="coerce")
    return (completion - start).dt.days.astype(float)


def resolve_target_series(df: pd.DataFrame, target_kind: str) -> pd.Series:
    """
    Raw target in days (float, NaN where undefined) before row filtering.

    ``primary_completion``: uses ``duration_days`` when present (preprocess output);
    otherwise ``primary_completion_date - start_date`` (same formula).

    ``post_primary_completion``: ``completion_date - primary_completion_date``.

    ``total_completion``: ``completion_date - start_date``.
    """
    if target_kind == "primary_completion":
        if TARGET_DURATION_COLUMN in df.columns:
            return pd.to_numeric(df[TARGET_DURATION_COLUMN], errors="coerce")
        return compute_days_to_primary_completion(df)
    if target_kind == "post_primary_completion":
        return compute_days_post_primary_completion(df)
    if target_kind == "total_completion":
        return compute_days_total_completion(df)
    raise ValueError(f"Unknown target_kind: {target_kind!r}")


def describe_target_kind(target_kind: str) -> tuple[str, str]:
    """Human-readable (title, formula) for reports."""
    if target_kind == "primary_completion":
        return (
            "primary_completion (start → primary_completion)",
            "primary_completion_date − start_date  [or column duration_days when present]",
        )
    if target_kind == "post_primary_completion":
        return (
            "post_primary_completion (primary → study completion)",
            "completion_date − primary_completion_date",
        )
    if target_kind == "total_completion":
        return (
            "total_completion (start → study completion)",
            "completion_date − start_date",
        )
    return (target_kind, "unknown")


def calculate_pct_deviation(
    actual: np.ndarray | float,
    predicted: np.ndarray | float,
    *,
    epsilon: float = _PCT_DEV_EPSILON,
) -> np.ndarray | float:
    """
    Percent deviation of actual duration vs predicted (symmetric in spirit to relative error on prediction scale):

        pct = (actual - predicted) / (predicted + epsilon) * 100

    Positive ⇒ actual longer than predicted ("late"). Vectorized for arrays.
    """
    a = np.asarray(actual, dtype=np.float64)
    p = np.asarray(predicted, dtype=np.float64)
    out = (a - p) / (p + epsilon) * 100.0
    if np.ndim(out) == 0:
        return float(out)
    return out


def make_late_flag(
    pct_deviation: np.ndarray | float,
    threshold_pct: float,
) -> np.ndarray | bool:
    """
    True where actual exceeded prediction by *more than* ``threshold_pct`` percent:

        late ⇔ pct_deviation > threshold_pct

    Scalar in → scalar bool; array in → bool array.
    """
    x = np.asarray(pct_deviation, dtype=np.float64)
    flags = x > threshold_pct
    if flags.ndim == 0:
        return bool(flags.item())
    return flags
