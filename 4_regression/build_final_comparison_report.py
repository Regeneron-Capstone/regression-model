#!/usr/bin/env python3
"""
Build a final comparison table + narrative report: frozen baseline vs staged planning-time system.

Reads:
  - Optional ``baseline_metadata.json`` (cohort sizes; frozen reference).
  - Regression report text files (``train_regression`` output) for primary/post_primary targets.
  - Late-risk classification report (test/val metrics).

Writes:
  - CSV (long-form metrics, labeled by target / feature policy / model family / scope / n).
  - Markdown report with cohort notes and baseline-vs-current callouts.

Usage (repo root, with PYTHONPATH=4_regression or from 4_regression):
  python 4_regression/build_final_comparison_report.py
  python 4_regression/build_final_comparison_report.py \\
    --frozen-regression results/frozen/regression_report_baseline_primary.txt \\
    --out-csv results/final_comparison_metrics.csv \\
    --out-md results/final_comparison_report.md
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

MODEL_FAMILY_REGRESSION = (
    "HistGradientBoostingRegressor + TransformedTargetRegressor(log1p/expm1)"
)
MODEL_FAMILY_CLASSIFIER = "HistGradientBoostingClassifier (max_iter=200, class_weight=balanced)"

_RE_TARGET = re.compile(r"target_kind:\s*(\S+)")
_RE_Y_TAG = re.compile(r"\[y=([^\]]+)\]")
_RE_POLICY = re.compile(r"feature_policy:\s*(\S+)")
_RE_SPLIT = re.compile(
    r"Split:\s*train=([\d,]+)\s+val=([\d,]+)\s+test=([\d,]+)",
    re.I,
)
# Reports use Unicode superscript ² in "R²="; allow ASCII ^2 fallback.
_RE_TEST_METRICS = re.compile(
    r"^\s*test\s*: RMSE=([\d,]+)\s*days\s*MAE=([\d,]+)\s*days\s*R\u00b2=([-\d.]+)",
)
_RE_TEST_METRICS_ASCII = re.compile(
    r"^\s*test\s*: RMSE=([\d,]+)\s*days\s*MAE=([\d,]+)\s*days\s*R\^2=([-\d.]+)",
    re.I,
)
_RE_SECTION_TITLE = re.compile(r"^MODEL\s+(.+)$", re.I)


def _parse_int_commas(s: str) -> int:
    return int(s.replace(",", "").strip())


def _parse_report_header(text: str) -> tuple[str, str]:
    """Infer (target_kind, feature_policy) from report preamble."""
    target = "primary_completion"
    policy = "baseline"
    m = _RE_TARGET.search(text)
    if m:
        target = m.group(1).strip()
    m = _RE_POLICY.search(text)
    if m:
        policy = m.group(1).strip()
    return target, policy


def parse_regression_report_file(path: Path, *, run_label: str) -> list[dict[str, Any]]:
    """Extract one row per MODEL section with pooled test metrics line."""
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    tgt, pol = _parse_report_header(text)

    rows: list[dict[str, Any]] = []
    # Reports often separate ``MODEL …`` from metric lines with a row of ``=====``; parse line-wise.
    current_title: str | None = None
    section_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("MODEL "):
            if current_title:
                row = _parse_model_section(
                    current_title,
                    section_lines,
                    run_label=run_label,
                    path=path,
                    tgt=tgt,
                    pol=pol,
                )
                if row:
                    rows.append(row)
            current_title = stripped
            section_lines = []
        elif current_title:
            section_lines.append(line)

    if current_title:
        row = _parse_model_section(
            current_title,
            section_lines,
            run_label=run_label,
            path=path,
            tgt=tgt,
            pol=pol,
        )
        if row:
            rows.append(row)
    return rows


def _parse_model_section(
    section_title: str,
    lines: list[str],
    *,
    run_label: str,
    path: Path,
    tgt: str,
    pol: str,
) -> dict[str, Any] | None:
    m_y = _RE_Y_TAG.search(section_title)
    if m_y:
        tgt = m_y.group(1).strip()
    scope = section_title[6:].strip() if section_title.startswith("MODEL ") else section_title

    test_rmse = test_mae = test_r2 = None
    n_train = n_val = n_test = None
    n_after_target: int | None = None

    for ln in lines:
        sm = _RE_SPLIT.search(ln)
        if sm:
            n_train = _parse_int_commas(sm.group(1))
            n_val = _parse_int_commas(sm.group(2))
            n_test = _parse_int_commas(sm.group(3))
        tm = _RE_TEST_METRICS.match(ln.strip()) or _RE_TEST_METRICS_ASCII.match(ln.strip())
        if tm:
            test_rmse = float(tm.group(1).replace(",", ""))
            test_mae = float(tm.group(2).replace(",", ""))
            test_r2 = float(tm.group(3))
        if "Rows after target filter" in ln and "finite" in ln:
            mm = re.search(r"n=([\d,]+)", ln)
            if mm:
                n_after_target = _parse_int_commas(mm.group(1))
        if "After target present:" in ln:
            mm = re.search(r"n=([\d,]+)", ln)
            if mm:
                n_after_target = _parse_int_commas(mm.group(1))

    if test_r2 is None:
        return None

    return {
        "run_label": run_label,
        "source_file": str(path.relative_to(PROJECT_ROOT)),
        "target": tgt,
        "feature_policy": pol,
        "model_family": MODEL_FAMILY_REGRESSION,
        "scope": scope,
        "metric_split": "test",
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "n_after_target_filter": n_after_target,
        "rmse_days": test_rmse,
        "mae_days": test_mae,
        "r2": test_r2,
        "precision": None,
        "recall": None,
        "f1": None,
        "roc_auc": None,
        "pr_auc": None,
    }


def parse_late_risk_report(path: Path, *, run_label: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    rows: list[dict[str, Any]] = []
    # Blocks starting with "Split: ..."
    for block in re.split(r"\n(?=Split:)", text):
        if "Split:" not in block:
            continue
        sm = re.search(r"Split:\s*([^\n]+)", block)
        if not sm:
            continue
        split_name = sm.group(1).strip()
        if "in-sample" in split_name:
            split_name = "train_in_sample"
        nm = re.search(r"n\s*=\s*([\d,]+)", block)
        n = _parse_int_commas(nm.group(1)) if nm else None

        def grab(pat: str) -> float | None:
            m = re.search(pat, block, re.I)
            return float(m.group(1)) if m else None

        rows.append(
            {
                "run_label": run_label,
                "source_file": str(path.relative_to(PROJECT_ROOT)),
                "target": "late_risk (total_completion quantile label)",
                "feature_policy": "strict_planning",
                "model_family": MODEL_FAMILY_CLASSIFIER,
                "scope": "late_risk_global",
                "metric_split": split_name,
                "n_train": n if split_name.startswith("train") else None,
                "n_val": n if split_name == "val" else None,
                "n_test": n if split_name == "test" else None,
                "n_after_target_filter": None,
                "rmse_days": None,
                "mae_days": None,
                "r2": None,
                "precision": grab(r"precision\s*=\s*([\d.]+)"),
                "recall": grab(r"recall\s*=\s*([\d.]+)"),
                "f1": grab(r"F1\s*=\s*([\d.]+)"),
                "roc_auc": grab(r"ROC-AUC\s*=\s*([\d.]+|n/a)"),
                "pr_auc": grab(r"PR-AUC\s*=\s*([\d.]+|n/a)"),
            }
        )
    # Clean n/a strings
    for r in rows:
        for k in ("roc_auc", "pr_auc"):
            if isinstance(r[k], str) and r[k] == "n/a":
                r[k] = None
    return rows


def load_baseline_metadata(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def build_comparison_dataframe(
    *,
    metadata_path: Path,
    frozen_regression: Path | None,
    primary_regression: Path,
    post_regression: Path,
    late_risk_report: Path,
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    rows: list[dict[str, Any]] = []

    meta = load_baseline_metadata(metadata_path)
    if meta:
        rows.append(
            {
                "run_label": "frozen_baseline_metadata_json",
                "source_file": str(metadata_path.relative_to(PROJECT_ROOT)),
                "target": meta.get("target_kind", "primary_completion"),
                "feature_policy": "baseline (reference snapshot)",
                "model_family": "metadata_only",
                "scope": "cohort_completed_trials",
                "metric_split": "n/a",
                "n_train": None,
                "n_val": None,
                "n_test": None,
                "n_after_target_filter": meta.get("completed_trials_total"),
                "rmse_days": None,
                "mae_days": None,
                "r2": None,
                "precision": None,
                "recall": None,
                "f1": None,
                "roc_auc": None,
                "pr_auc": None,
            }
        )

    if frozen_regression and frozen_regression.exists():
        rows.extend(parse_regression_report_file(frozen_regression, run_label="frozen_baseline_regression"))

    rows.extend(parse_regression_report_file(primary_regression, run_label="current_baseline_primary"))
    rows.extend(parse_regression_report_file(post_regression, run_label="staged_post_primary_strict"))
    rows.extend(parse_late_risk_report(late_risk_report, run_label="late_risk_classifier"))

    df = pd.DataFrame(rows)
    return df, meta


def _md_table(df: pd.DataFrame, columns: list[str]) -> str:
    """Markdown pipe table without requiring the ``tabulate`` package."""
    sub = df[columns].copy()
    sub = sub.astype(object).where(pd.notna(sub), "")
    header = "| " + " | ".join(str(c) for c in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, sep]
    for _, row in sub.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in columns) + " |")
    return "\n".join(lines)


def write_markdown_report(
    path: Path,
    df: pd.DataFrame,
    meta: dict[str, Any] | None,
    *,
    metadata_path: Path,
    frozen_path: Path | None,
    primary_path: Path,
    post_path: Path,
    late_path: Path,
) -> None:
    lines: list[str] = []
    lines.append("# Final comparison: baseline vs staged planning-time system")
    lines.append("")
    lines.append("This report aggregates **regression** metrics (duration models) and **classification** metrics ")
    lines.append("(late-risk) with explicit **target**, **feature policy**, **model family**, and **sample sizes**.")
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    lines.append("| Artifact | Path |")
    lines.append("|----------|------|")
    lines.append(f"| Baseline metadata (optional) | `{metadata_path}` |")
    if frozen_path:
        lines.append(f"| Frozen baseline regression (optional) | `{frozen_path}` |")
    lines.append(f"| Current primary baseline regression | `{primary_path}` |")
    lines.append(f"| Post-primary strict planning regression | `{post_path}` |")
    lines.append(f"| Late-risk classification | `{late_path}` |")
    lines.append("")

    if meta:
        lines.append("## Frozen baseline cohort snapshot")
        lines.append("")
        lines.append(f"- **Completed trials (total):** {meta.get('completed_trials_total', 'n/a'):,}")
        if "completed_trials_by_phase_label" in meta:
            lines.append("- **By phase label:**")
            for ph, n in meta["completed_trials_by_phase_label"].items():
                lines.append(f"  - {ph}: {n:,}")
        lines.append(f"- **Metadata generated (UTC):** {meta.get('generated_at_utc', 'n/a')}")
        lines.append("")

    lines.append("## Regression: dedicated models (test split)")
    lines.append("")
    reg = df[
        (df["model_family"] == MODEL_FAMILY_REGRESSION)
        & (df["scope"].str.startswith("dedicated", na=False))
        & (df["metric_split"] == "test")
    ].copy()
    if not reg.empty:
        cols = [
            "run_label",
            "target",
            "feature_policy",
            "scope",
            "n_test",
            "rmse_days",
            "mae_days",
            "r2",
        ]
        lines.append(_md_table(reg, cols))
    else:
        lines.append("_No dedicated-model rows parsed (check report paths)._")
    lines.append("")

    lines.append("## Regression: joint models (pooled test line)")
    lines.append("")
    joint = df[
        (df["model_family"] == MODEL_FAMILY_REGRESSION)
        & (df["scope"].str.contains("joint", case=False, na=False))
        & (df["metric_split"] == "test")
    ].copy()
    if not joint.empty:
        lines.append(_md_table(joint, ["run_label", "target", "feature_policy", "scope", "n_test", "r2"]))
    else:
        lines.append("_No joint-model rows found._")
    lines.append("")

    lines.append("## Classification: late risk (strict planning features)")
    lines.append("")
    clsf = df[df["model_family"] == MODEL_FAMILY_CLASSIFIER].copy()
    if not clsf.empty:
        show = clsf[clsf["metric_split"].isin(["val", "test"])].copy()
        show["n"] = show["n_val"].combine_first(show["n_test"])
        cols = [
            "run_label",
            "metric_split",
            "n",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "pr_auc",
        ]
        lines.append(_md_table(show, cols))
    else:
        lines.append("_No classification rows (missing late-risk report?)._")
    lines.append("")

    lines.append("## Baseline vs current (primary_completion, dedicated test)")
    lines.append("")
    fr = df[(df["run_label"] == "frozen_baseline_regression") & (df["scope"].str.startswith("dedicated", na=False))]
    cr = df[(df["run_label"] == "current_baseline_primary") & (df["scope"].str.startswith("dedicated", na=False))]
    if not fr.empty and not cr.empty:
        merged = fr[["scope", "r2", "rmse_days"]].merge(
            cr[["scope", "r2", "rmse_days"]],
            on="scope",
            suffixes=("_frozen", "_current"),
        )
        merged["delta_r2"] = merged["r2_current"] - merged["r2_frozen"]
        lines.append(_md_table(merged, list(merged.columns)))
    else:
        lines.append(
            "_Supply `--frozen-regression` pointing to a saved baseline report to enable side-by-side "
            "dedicated-model deltas. Without it, only `current_baseline_primary` rows appear._"
        )
    lines.append("")

    lines.append("## Model families (fixed across runs)")
    lines.append("")
    lines.append(f"- **Regression:** `{MODEL_FAMILY_REGRESSION}`")
    lines.append(f"- **Late risk:** `{MODEL_FAMILY_CLASSIFIER}`")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Build final baseline vs staged comparison CSV + Markdown")
    p.add_argument(
        "--metadata",
        type=Path,
        default=RESULTS_DIR / "baseline_metadata.json",
        help="Frozen baseline_metadata.json (optional)",
    )
    p.add_argument(
        "--frozen-regression",
        type=Path,
        default=None,
        help="Saved frozen primary baseline regression_report (optional; enables delta table)",
    )
    p.add_argument(
        "--primary-regression",
        type=Path,
        default=RESULTS_DIR / "regression_report.txt",
        help="Current primary baseline report (default: results/regression_report.txt)",
    )
    p.add_argument(
        "--post-regression",
        type=Path,
        default=RESULTS_DIR / "regression_report_post_primary_completion_strict_planning.txt",
        help="Post-primary strict planning report",
    )
    p.add_argument(
        "--late-risk-report",
        type=Path,
        default=RESULTS_DIR / "late_risk_classification_report.txt",
        help="Late-risk evaluation report",
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        default=RESULTS_DIR / "final_comparison_metrics.csv",
        help="Output CSV path",
    )
    p.add_argument(
        "--out-md",
        type=Path,
        default=RESULTS_DIR / "final_comparison_report.md",
        help="Output Markdown path",
    )
    p.add_argument(
        "--out-txt",
        type=Path,
        default=None,
        help="Optional plain-text copy (same content as Markdown without tables rendered)",
    )
    args = p.parse_args()

    df, meta = build_comparison_dataframe(
        metadata_path=args.metadata.expanduser().resolve(),
        frozen_regression=args.frozen_regression.expanduser().resolve() if args.frozen_regression else None,
        primary_regression=args.primary_regression.expanduser().resolve(),
        post_regression=args.post_regression.expanduser().resolve(),
        late_risk_report=args.late_risk_report.expanduser().resolve(),
    )

    out_csv = args.out_csv.expanduser().resolve()
    out_md = args.out_md.expanduser().resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    frozen = args.frozen_regression.expanduser().resolve() if args.frozen_regression else None
    write_markdown_report(
        out_md,
        df,
        meta,
        metadata_path=args.metadata.expanduser().resolve(),
        frozen_path=frozen,
        primary_path=args.primary_regression.expanduser().resolve(),
        post_path=args.post_regression.expanduser().resolve(),
        late_path=args.late_risk_report.expanduser().resolve(),
    )

    if args.out_txt:
        out_txt = args.out_txt.expanduser().resolve()
        out_txt.write_text(out_md.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"Wrote {out_txt}")

    print(f"Wrote {out_csv}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
