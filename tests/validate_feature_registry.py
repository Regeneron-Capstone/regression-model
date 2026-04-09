"""
Tests for 4_regression/feature_registry.py (run without pytest: python tests/validate_feature_registry.py).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "4_regression"))

from feature_registry import (  # noqa: E402
    get_feature_policy,
    validate_no_leakage,
    validate_strict_planning_feature_set,
)


def test_validate_no_leakage_passes_without_forbidden() -> None:
    policy = get_feature_policy("strict_planning")
    validate_no_leakage(["enrollment", "phase", "category"], policy.forbidden)


def test_validate_no_leakage_fails_on_forbidden_start_year() -> None:
    policy = get_feature_policy("strict_planning")
    try:
        validate_no_leakage(["enrollment", "start_year", "phase"], policy.forbidden)
    except ValueError as e:
        assert "start_year" in str(e)
        assert "forbidden" in str(e).lower() or "leakage" in str(e).lower()
        return
    raise AssertionError("expected ValueError when forbidden column present")


def test_validate_no_leakage_fails_on_site_footprint() -> None:
    policy = get_feature_policy("strict_planning")
    try:
        validate_no_leakage(["number_of_facilities"], policy.forbidden)
    except ValueError:
        return
    raise AssertionError("expected ValueError for number_of_facilities")


def test_get_feature_policy_unknown() -> None:
    try:
        get_feature_policy("nonexistent_policy")  # type: ignore[arg-type]
    except KeyError:
        return
    raise AssertionError("expected KeyError")


def test_strict_allowlist_accepts_planning_safe_bundle() -> None:
    """A bundle matching STRICT_PLANNING_ALLOWED should pass full strict check."""
    policy = get_feature_policy("strict_planning")
    validate_strict_planning_feature_set(sorted(policy.allowed))


def test_strict_allowlist_rejects_unknown_column() -> None:
    policy = get_feature_policy("strict_planning")
    try:
        validate_strict_planning_feature_set(list(policy.allowed) + ["mystery_feature"])
    except ValueError as e:
        assert "mystery_feature" in str(e)
        return
    raise AssertionError("expected ValueError for unknown column")


def main() -> int:
    test_validate_no_leakage_passes_without_forbidden()
    test_validate_no_leakage_fails_on_forbidden_start_year()
    test_validate_no_leakage_fails_on_site_footprint()
    test_get_feature_policy_unknown()
    test_strict_allowlist_accepts_planning_safe_bundle()
    test_strict_allowlist_rejects_unknown_column()
    print("tests/validate_feature_registry.py: all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
