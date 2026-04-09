"""
Structured regression and deviation-style metrics for trial duration models.

Core metrics (RMSE, MAE, R²) match ``sklearn`` definitions used in ``train_regression``.
Percentage deviation metrics follow ``targets.calculate_pct_deviation`` and the reporting
style in ``5_deviation/deviation_analysis.py`` (MAPE = mean(|pct_dev|), pandas-style std).
"""
from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.compose import TransformedTargetRegressor

from targets import calculate_pct_deviation


def core_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """RMSE, MAE, R² on arrays (float64)."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def mae_days(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error in original units (days)."""
    return float(mean_absolute_error(np.asarray(y_true, dtype=np.float64), np.asarray(y_pred, dtype=np.float64)))


def deviation_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    quantiles: tuple[float, ...] = (0.25, 0.5, 0.75, 0.9),
    bands: tuple[int, ...] = (10, 20, 30),
) -> dict[str, float]:
    """
    Metrics derived from percent deviation: (actual - pred) / (pred + eps) * 100.

    - ``mape``: mean absolute pct deviation (same convention as deviation_analysis printout).
    - ``std_pct_deviation``: pandas-compatible sample std (ddof=1) on pct series.
    - ``pct_q_XX``: quantiles of the pct deviation distribution (XX = int(100*q)).
    - ``within_pm_XX_pct``: % of samples with |pct_dev| <= XX.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    pct = np.asarray(calculate_pct_deviation(y_true, y_pred), dtype=np.float64).ravel()
    ser = pd.Series(pct)

    out: dict[str, float] = {
        "mape": float(np.mean(np.abs(pct))),
        "mean_pct_deviation": float(np.mean(pct)),
        "median_pct_deviation": float(np.median(pct)),
        "std_pct_deviation": float(ser.std()) if len(ser) > 0 else float("nan"),
    }
    for q in quantiles:
        key = f"pct_q_{int(round(q * 100))}"
        out[key] = float(ser.quantile(q)) if len(ser) > 0 else float("nan")
    for b in bands:
        out[f"within_pm_{b}_pct"] = float(np.mean(np.abs(pct) <= b) * 100.0) if len(pct) > 0 else float("nan")
    return out


def full_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    split_name: str | None = None,
    include_deviation: bool = True,
) -> dict[str, Any]:
    """Merge core regression metrics with optional deviation block."""
    out: dict[str, Any] = dict(core_regression_metrics(y_true, y_pred))
    if split_name is not None:
        out["set"] = split_name
    if include_deviation:
        out.update(deviation_metrics(y_true, y_pred))
    return out


def evaluate_sklearn_split(
    split_name: str,
    model: TransformedTargetRegressor,
    X: np.ndarray,
    y: np.ndarray,
    *,
    include_deviation: bool = False,
) -> dict[str, Any]:
    """
    Predict with ``model`` and return metrics dict (always includes rmse, mae, r2, set).

    Default ``include_deviation=False`` matches historical train_regression report density.
    """
    y_pred = model.predict(X)
    return full_metrics(y, y_pred, split_name=split_name, include_deviation=include_deviation)


def evaluate_subset(
    model: TransformedTargetRegressor,
    X: np.ndarray,
    y: np.ndarray,
) -> dict[str, float]:
    """RMSE/MAE/R² plus row count (for joint-model per-phase test slices)."""
    y_pred = model.predict(X)
    m = core_regression_metrics(y, y_pred)
    m["n"] = int(len(y))
    return m


def metrics_report_line(split_metrics: dict[str, Any]) -> str:
    """Single train/val/test line as in regression_report.txt (with 'days' suffix)."""
    return (
        f"  {split_metrics['set']:5}: RMSE={split_metrics['rmse']:,.0f} days  "
        f"MAE={split_metrics['mae']:,.0f} days  R²={split_metrics['r2']:.4f}"
    )


def joint_subset_report_line(phase: str, d: dict[str, float]) -> str:
    """Per-phase joint test line (no 'days' after RMSE in historical table)."""
    return (
        f"  test ({phase} subset): n={d['n']:,}  "
        f"RMSE={d['rmse']:,.0f}  MAE={d['mae']:,.0f}  R²={d['r2']:.4f}"
    )


def mixed_cohort_test_line(n_test: int, te: dict[str, float]) -> str:
    """Indented test line for mixed-phase baseline block."""
    return (
        f"    test: n={n_test:,}  RMSE={te['rmse']:,.0f}  MAE={te['mae']:,.0f}  R²={te['r2']:.4f}"
    )


def evaluations_to_dataframe(rows: Iterable[dict[str, Any]]) -> pd.DataFrame:
    """Stack metric dicts (e.g. per-split) into a DataFrame."""
    return pd.DataFrame(list(rows))


def deviation_summary_to_dataframe(summary: dict[str, Any]) -> pd.DataFrame:
    """One-row DataFrame from a flat deviation_metrics / full_metrics dict."""
    return pd.DataFrame([summary])


def format_deviation_summary_report(
    df: pd.DataFrame,
    *,
    phase_order: Sequence[str],
    late_threshold_pct: float,
    phase_col: str = "phase",
    pct_col: str = "pct_deviation",
    late_flag_col: str = "late_flag",
    abs_error_col: str = "abs_error_days",
    category_col: str | None = "category",
    title: str = "PREDICTION DEVIATION ANALYSIS",
    header_extra: Sequence[str] = (),
    group_col: str | None = None,
) -> str:
    """
    Text report matching historical ``baseline_deviation`` style: per-phase (and optional
    ``group_col``) MAPE bands, late flags, and top categories.
    """
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append(title)
    lines.append(f"Late threshold: >{late_threshold_pct:.0f}% over prediction")
    for h in header_extra:
        lines.append(h)
    lines.append(f"Total rows in table: {len(df):,}")
    lines.append("=" * 70)

    def _phase_blocks(sub: pd.DataFrame, phase_key: str) -> None:
        for phase in phase_order:
            phase_df = sub[sub[phase_col] == phase]
            if phase_df.empty:
                continue
            pct_dev = phase_df[pct_col]
            mape = float(np.mean(np.abs(pct_dev)))
            mae_days = float(phase_df[abs_error_col].mean())
            n_late = int(phase_df[late_flag_col].sum())
            within_10 = float((np.abs(pct_dev) <= 10).mean() * 100)
            within_20 = float((np.abs(pct_dev) <= 20).mean() * 100)
            within_30 = float((np.abs(pct_dev) <= 30).mean() * 100)
            lines.append("")
            lines.append(f"{phase_key}PHASE: {phase}")
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
            lr = float(phase_df[late_flag_col].mean() * 100)
            lines.append(f"  Trials flagged as late:   {n_late:,} ({lr:.1f}%)")
            lines.append("")
            lines.append("  Accuracy bands:")
            lines.append(f"    Within ±10%:            {within_10:.1f}%")
            lines.append(f"    Within ±20%:            {within_20:.1f}%")
            lines.append(f"    Within ±30%:            {within_30:.1f}%")
            lines.append("=" * 70)

    if group_col is not None and group_col in df.columns:
        for g in sorted(df[group_col].dropna().unique(), key=str):
            lines.append("")
            lines.append(f"--- {group_col}={g!r} ---")
            _phase_blocks(df[df[group_col] == g], phase_key="")
    else:
        _phase_blocks(df, phase_key="")

    cat_col = category_col
    if (
        cat_col
        and cat_col in df.columns
        and df[cat_col].notna().any()
        and (df[cat_col].astype(str) != "").any()
    ):
        lines.append("")
        lines.append("PERFORMANCE BY DISEASE CATEGORY (Top 10 by trial count)")
        lines.append("=" * 70)
        cat_stats = (
            df.groupby(cat_col)
            .agg(
                N_trials=(pct_col, "count"),
                Mean_pct_dev=(pct_col, "mean"),
                MAPE=(pct_col, lambda x: np.mean(np.abs(x))),
                Late_rate=(late_flag_col, "mean"),
            )
            .sort_values("N_trials", ascending=False)
            .head(10)
            .round(2)
        )
        cat_stats["Late_rate"] = (cat_stats["Late_rate"] * 100).round(1)
        lines.append(cat_stats.to_string())
        lines.append("=" * 70)

    return "\n".join(lines)
