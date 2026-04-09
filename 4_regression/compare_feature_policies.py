#!/usr/bin/env python3
"""
Compare ``assemble_feature_matrix`` under ``baseline`` vs ``strict_planning`` policies.

Prints row count, feature dimension, which high-level groups contributed columns,
and whether any strict-forbidden logical columns remain.

Usage (from repo root, with clean_data + raw_data present):
  python 4_regression/compare_feature_policies.py
  python 4_regression/compare_feature_policies.py --max-rows 8000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "4_regression"))
sys.path.insert(0, str(PROJECT_ROOT / "3_preprocessing"))

import train_regression as tr  # noqa: E402
from feature_registry import get_feature_policy  # noqa: E402
from features import assemble_feature_matrix  # noqa: E402


def _group_summary(art: dict) -> list[str]:
    lines = []
    if art.get("phase_encoder") is not None:
        lines.append("phase_one_hot")
    lines.append("category_one_hot")
    if art.get("mesh_encoder") is not None:
        lines.append("mesh_term_one_hot")
    if art.get("int_encoder") is not None:
        lines.append("intervention_type_one_hot")
    if art.get("elig_feature_names"):
        lines.append(f"eligibility ({len(art['elig_feature_names'])} expanded)")
    if art.get("criteria_text_feature_names"):
        lines.append(f"criteria_text ({len(art['criteria_text_feature_names'])} cols)")
    if art.get("site_feature_names"):
        lines.append(f"site_footprint ({len(art['site_feature_names'])} cols)")
    if art.get("design_feature_names"):
        lines.append(f"design ({len(art['design_feature_names'])} expanded)")
    if art.get("do_feature_names"):
        lines.append(f"design_outcomes ({len(art['do_feature_names'])} cols)")
    if art.get("arm_feature_names"):
        lines.append(f"arm_intervention ({len(art['arm_feature_names'])} cols)")
    if art.get("numeric_feature_names"):
        lines.append(f"numeric_tail ({', '.join(art['numeric_feature_names'])})")
    return lines


def _forbidden_hits(art: dict) -> list[str]:
    fp = get_feature_policy("strict_planning")
    logical = set(art.get("logical_source_columns", []))
    return sorted(logical & fp.forbidden)


def run_policy(df, policy: str, prep_kw: dict) -> None:
    X, y, _, art = assemble_feature_matrix(df, **prep_kw, policy=policy)  # type: ignore[arg-type]
    print(f"\n=== policy={policy!r} | target_kind={art.get('target_kind', '?')!r} ===")
    print(f"  rows (after target dropna): {len(y):,}")
    print(f"  feature columns (X width): {X.shape[1]:,}")
    print("  top-level groups:")
    for g in _group_summary(art):
        print(f"    - {g}")
    bad = _forbidden_hits(art)
    print(f"  forbidden logical columns present (vs strict_planning registry): {bad or 'none'}")
    print(f"  logical_source_columns ({len(art.get('logical_source_columns', []))}): {art.get('logical_source_columns')}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare feature policies on a loaded cohort slice")
    parser.add_argument("--max-rows", type=int, default=None, help="Cap rows after COMPLETED filter (default: all)")
    args = parser.parse_args()

    print("Loading joined data (same as train_regression)...")
    df = tr.load_and_join(
        eligibility_columns=tr.KEPT_ELIGIBILITY,
        site_footprint_columns=tr.KEPT_SITE_FOOTPRINT,
        design_columns=tr.KEPT_DESIGN,
        arm_intervention_columns=tr.KEPT_ARM_INTERVENTION,
        design_outcomes_columns=tr.KEPT_DESIGN_OUTCOMES,
    )
    df = df[df["overall_status"] == "COMPLETED"].copy()
    if args.max_rows is not None:
        df = df.head(args.max_rows).copy()
    print(f"COMPLETED cohort slice: {len(df):,} rows")

    prep_kw = dict(
        eligibility_columns=tr.KEPT_ELIGIBILITY,
        eligibility_criteria_text_columns=tr.KEPT_ELIGIBILITY_CRITERIA_TEXT,
        site_footprint_columns=tr.KEPT_SITE_FOOTPRINT,
        design_columns=tr.KEPT_DESIGN,
        arm_intervention_columns=tr.KEPT_ARM_INTERVENTION,
        design_outcomes_columns=tr.KEPT_DESIGN_OUTCOMES,
        encode_phase=False,
    )

    run_policy(df, "baseline", prep_kw)
    run_policy(df, "strict_planning", prep_kw)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
