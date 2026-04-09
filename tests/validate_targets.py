"""
Lightweight checks for 4_regression/targets.py (no pytest required).

Run from repo root:
  python tests/validate_targets.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REG = ROOT / "4_regression"
sys.path.insert(0, str(REG))

from targets import (  # noqa: E402
    TARGET_DURATION_COLUMN,
    calculate_pct_deviation,
    compute_days_post_primary_completion,
    compute_days_to_primary_completion,
    compute_days_total_completion,
    make_late_flag,
    resolve_target_series,
)


def main() -> int:
    toy = pd.DataFrame(
        {
            "start_date": ["2020-01-01", "2019-06-15"],
            "primary_completion_date": ["2020-01-11", "2019-08-14"],
            "completion_date": ["2020-02-01", "2019-09-30"],
        }
    )

    to_primary = compute_days_to_primary_completion(toy)
    assert list(to_primary) == [10.0, 60.0], to_primary.tolist()

    post = compute_days_post_primary_completion(toy)
    assert list(post) == [21.0, 47.0], post.tolist()

    total = compute_days_total_completion(toy)
    assert list(total) == [31.0, 107.0], total.tolist()

    # Match preprocess-style row: duration_days == primary - start
    assert float(to_primary.iloc[0]) == 10.0

    r1 = resolve_target_series(toy.assign(**{TARGET_DURATION_COLUMN: [10.0, 60.0]}), "primary_completion")
    assert list(r1) == [10.0, 60.0]
    r2 = resolve_target_series(toy, "post_primary_completion")
    assert list(r2) == [21.0, 47.0]
    r3 = resolve_target_series(toy, "total_completion")
    assert list(r3) == [31.0, 107.0]

    actual = np.array([120.0, 80.0])
    pred = np.array([100.0, 100.0])
    pct = calculate_pct_deviation(actual, pred)
    assert np.allclose(pct, [20.0, -20.0]), pct

    assert abs(calculate_pct_deviation(120.0, 100.0) - 20.0) < 1e-9

    assert make_late_flag(20.0, 20.0) is False
    assert make_late_flag(20.0000001, 20.0) is True
    flags = make_late_flag(np.array([10.0, 25.0]), 20.0)
    assert list(np.asarray(flags).tolist()) == [False, True]

    # NaT handling: should yield NaN days
    bad = pd.DataFrame(
        {
            "start_date": ["2020-01-01", None],
            "primary_completion_date": ["2020-01-02", "2020-01-02"],
            "completion_date": ["2020-01-03", "2020-01-03"],
        }
    )
    s = compute_days_to_primary_completion(bad)
    assert pd.notna(s.iloc[0]) and np.isnan(s.iloc[1])

    print("tests/validate_targets.py: all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
