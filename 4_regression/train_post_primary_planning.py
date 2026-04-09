#!/usr/bin/env python3
"""
Train the planning-safe post-primary-completion duration model (parallel to baseline primary).

- Target: completion_date − primary_completion_date (days).
- Features: strict_planning policy only (no start_year, no site-footprint fields).
- Model: HistGradientBoostingRegressor + TransformedTargetRegressor(log1p / expm1), same as train_regression.
- Report: results/regression_report_post_primary_completion_strict_planning.txt

Usage (from repo root):
  python 4_regression/train_post_primary_planning.py

Equivalent CLI:
  python 4_regression/train_regression.py \\
    --target post_primary_completion \\
    --feature-policy strict_planning \\
    --report results/regression_report_post_primary_completion_strict_planning.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import train_regression as tr  # noqa: E402

REPORT_NAME = "regression_report_post_primary_completion_strict_planning.txt"


def main() -> None:
    tr.run_training(
        "post_primary_completion",
        feature_policy="strict_planning",
        report_path=tr.RESULTS_DIR / REPORT_NAME,
        random_state=42,
    )


if __name__ == "__main__":
    main()
