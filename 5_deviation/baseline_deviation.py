"""
Backward-compatible entry point for primary-completion deviation analysis.

Equivalent to:
  python 5_deviation/deviation_analysis.py --target primary_completion

For other targets and combined staged predictions, use ``deviation_analysis.py``.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_REGRESSION_DIR = PROJECT_ROOT / "4_regression"
if str(_REGRESSION_DIR) not in sys.path:
    sys.path.insert(0, str(_REGRESSION_DIR))

from deviation_analysis import run_analysis  # noqa: E402

LATE_THRESHOLD_PCT: float = 20.0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main(threshold: float = LATE_THRESHOLD_PCT) -> None:
    run_analysis(
        target="primary_completion",
        threshold_pct=threshold,
        random_state=42,
        combined_csv=None,
        output_csv=None,
        output_summary=None,
        splits=("test",),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trial duration prediction deviation (primary target; legacy CLI)")
    parser.add_argument(
        "--threshold",
        type=float,
        default=LATE_THRESHOLD_PCT,
        help="Percent deviation above prediction to flag as late (default: %(default)s)",
    )
    args = parser.parse_args()
    main(threshold=args.threshold)
